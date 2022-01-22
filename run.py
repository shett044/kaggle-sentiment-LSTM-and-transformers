import math
import time
from collections import OrderedDict
from datetime import date
from importlib import reload

import numpy as np
import torch
from path import Path
from sklearn.metrics import accuracy_score
from sklearn.utils import compute_class_weight
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.common import utils as common_utils
from src.data import dataset_util
from src.data import movie_sentimement_dataset
from src.features import vocab_utils
from src.models import senti_transformer, senti_bilstm

reload(dataset_util)
reload(vocab_utils)
reload(movie_sentimement_dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds_typ = dataset_util.DSType
SPECIALS = ['<unk>', '<pad>', '<sos>', '<eos>']
SPECIAL_SYMBOL = OrderedDict(zip(SPECIALS, range(len(SPECIALS))))
VOCAB_REFRESH = False

EMBED_SIZE = 256
HIDDEN_SIZE = 256
NLAYER = 1
NHEAD = 2
BATCH_SIZE = 128
CLIP_GRAD_NORM = 0.25
LR = 0.01
NUM_EPOCH = 30
LOG_INTERVAL = 100
PATIENCE = 2
LR_DECAY = .5
model_dir = Path('models')
MODEL_TYPE = 'Bi-LSTM'
MODEL_NAME = f'{MODEL_TYPE}_EP-{NUM_EPOCH}_EM-{EMBED_SIZE}_HS-{HIDDEN_SIZE}_NL-{NLAYER}_NH-{NHEAD}_BS-{BATCH_SIZE}'
MODEL_SAVE_PATH = model_dir.joinpath(f'{MODEL_NAME}.pth')

movie_senti_ds = dataset_util.DataSpace(
    train_filename='data/raw/train.tsv.zip',
    test_filename='data/raw/test.tsv.zip',
    get_x=lambda df: df['Phrase'],
    get_y=lambda df: df['Sentiment'] if 'Sentiment' in df else None,
    sep='\t', compression='zip'
)

train_val_y = movie_senti_ds.get_XY_split(ds_typ.TRAINVAL)[1]
NLABEL = train_val_y.nunique()
wt_y = compute_class_weight("balanced", classes=sorted(np.unique(train_val_y)), y=train_val_y)
movie_senti_ds.gen_validation_index(val_size=.2, stratify=train_val_y, shuffle=True)

if VOCAB_REFRESH:
    vocab = vocab_utils.VocabFactory(specials=SPECIALS)
    vocab.build_vocab(d_iter=movie_senti_ds.yield_XY_split(ds_typ.TRAINVAL))
else:
    vocab = vocab_utils.VocabFactory.load_vocab(model_dir.joinpath('vocab_obj.pth'), SPECIALS)

NTOKENS = vocab.get_vocab_size()


def to_torch_tensor(x, device):
    return torch.tensor([SPECIAL_SYMBOL['<sos>']] + x + [SPECIAL_SYMBOL['<eos>']], device=device)


sent_transformer = common_utils.sequential_transforms(vocab.transform, lambda x: to_torch_tensor(x, device=device))
batch_transformer = lambda x: pad_sequence(x, padding_value=SPECIAL_SYMBOL['<pad>'])
collate_fn = lambda x: movie_sentimement_dataset.collate_batch(x, sent_transformer,
                                                               batch_transformer)
train_iter = movie_sentimement_dataset.MovieSentimentDataset(*movie_senti_ds.get_XY_split(ds_typ.TRAIN),
                                                             is_test=False)
val_iter = movie_sentimement_dataset.MovieSentimentDataset(*movie_senti_ds.get_XY_split(ds_typ.VAL),
                                                           is_test=False)
test_iter = movie_sentimement_dataset.MovieSentimentDataset(*movie_senti_ds.get_XY_split(ds_typ.TEST), is_test=True)

# train_sample_wt = train_iter.get_sample_weight(wt_y)

# Performance worsened
# weighted_sampler = WeightedRandomSampler(
#     weights=train_sample_wt, num_samples=len(train_sample_wt), replacement=True
# )

train_loader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True,
                          # sampler=weighted_sampler
                          )

val_loader = DataLoader(val_iter, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

test_loader = DataLoader(test_iter, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

if MODEL_TYPE == 'Bi-LSTM':
    model = senti_bilstm.SentimentBiLSTM(ntoken=NTOKENS, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
                                         num_layers=NLAYER, dropout_rate=.2).to(device)
else:
    model = senti_transformer.SentimentTransformerModel(ntoken=NTOKENS, ninp=EMBED_SIZE, nhead=NHEAD,
                                                        nhid=HIDDEN_SIZE, nlayers=NLAYER, nlabel=NLABEL).to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=1)
criterion_wt = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(wt_y))

optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
uniform_init = 0.1
for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)
#
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     "max",
#     patience=PATIENCE,
#     factor=LR_DECAY,
#     verbose=True
# )

from src.models import custom_scheduler

scheduler = custom_scheduler.CustomSchedulerLRPlateau(optimizer,
                                                      "max",
                                                      patience=PATIENCE,
                                                      factor=LR_DECAY,
                                                      verbose=True)


def get_ypred(ypred):
    with torch.no_grad():
        y_pred_softmax = torch.log_softmax(ypred, dim=-1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=-1)
    return y_pred_tags


def multi_acc(y_test, y_pred, class_weight=None):
    if class_weight is None:
        class_weight_sample = np.ones_like(y_test)
    else:
        class_weight_sample = class_weight[y_test]

    acc = accuracy_score(y_test, y_pred, sample_weight=class_weight_sample)
    return acc


def evaluate(data_source, model):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        y = []
        ypred = []
        for idx, data, targets, src_len in data_source:
            output = model(data, src_len)
            total_loss += len(data) * criterion(output, targets).item()
            ypred.append(get_ypred(output))
            y.append(targets)
        return {"Loss": total_loss / (len(data_source) - 1), "Accuracy": multi_acc(torch.cat(y), torch.cat(ypred))}


def submit_df(data_source, model):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    with torch.no_grad():
        ypred = []
        idx = []
        for i, data, targets, src_len in data_source:
            output = model(data, src_len)
            ypred.append(get_ypred(output))
            idx.append(i)
        return {"pred": torch.cat(ypred).numpy(), "index": torch.cat(idx).numpy()}


# torch.autograd.set_detect_anomaly(True)

def train(epoch, model, optimizer, new_lr=LR):
    # Turn on training mode which enables dropout.
    epoch_loss = total_loss = 0.
    model.train()
    start_time = time.time()

    for batch, (idx, data, targets, seq_mask) in enumerate(train_loader):
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        output = model(data, seq_mask)
        optimizer.zero_grad()
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()
        if optimizer.state_dict()['state'][0]['exp_avg'].isnan().sum().item() > 0:
            print("optimizer lol")
        # for p in model.parameters():
        #     p.data.add_(p.grad, alpha=-LR)

        total_loss += loss.item()
        if batch % LOG_INTERVAL == 0 and batch > 0:
            cur_loss = total_loss / LOG_INTERVAL
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | train_acc {:8.2f}'.format(
                epoch, batch, len(train_loader), new_lr,
                elapsed * 1000 / LOG_INTERVAL, cur_loss, multi_acc(targets, get_ypred(output))
            )
            )
            epoch_loss += total_loss
            total_loss = 0
            start_time = time.time()
    return epoch_loss


best_val_loss = None
patience = 0


def save_checkpoint(model, optimizer, scheduler, epoch=0):
    with open(MODEL_SAVE_PATH, 'wb') as f:
        print(f"Saving the best model : {MODEL_SAVE_PATH}")
        torch.save({
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": epoch
        }, f)

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, NUM_EPOCH + 1):
        epoch_start_time = time.time()
        train_epoch_loss = train(epoch, model, optimizer, (getattr(scheduler, '_last_lr', None) or [LR])[0])
        val_loss, val_acc = evaluate(val_loader, model).values()
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | val_acc {:5.2f}|'
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, 100 * val_acc, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        scheduler.step(lambda x: save_checkpoint(model, optimizer, scheduler, epoch), val_acc)
        # if not best_val_loss or val_acc < best_val_loss:
        #     with open(MODEL_SAVE_PATH, 'wb') as f:
        #         print(f"Saving the best model : {MODEL_SAVE_PATH}")
        #         torch.save(model, f)
        #     best_val_loss = val_acc
        # else:
        #     if patience < PATIENCE:
        #         patience += 1
        #         continue
        #     print(f"{best_val_loss=}")
        #     # Anneal the learning rate if no improvement has been seen in the validation dataset.
        #     LR *= LR_DECAY
        #     patience = 0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(MODEL_SAVE_PATH, 'rb') as f:
    checkpoint = torch.load(f)
    model = checkpoint['model']
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.

# Run on test data.

with torch.no_grad():
    print('=' * 89)
    val_loss, val_acc = evaluate(val_loader, model).values()
    print('| End of training |  valid loss {:5.2f} | val_acc {:5.2f}'.format(
        val_loss, 100 * val_acc))
    res = submit_df(test_loader, model)
    test_df = movie_senti_ds.get_df_split(ds_typ.TEST)
    test_df.loc[res['index'], 'Sentiment'] = res['pred']
    assert test_df['Sentiment'].isnull().sum() == 0
    test_df[['PhraseId', 'Sentiment']].to_csv(f'data/submission/submit_{MODEL_NAME}_{date.today()}.csv', index=False)

    print(f"Submission file: data/submission/submit_{MODEL_NAME}_{date.today()}.csv")
    print('=' * 89)
