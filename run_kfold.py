import math
import re
import time
from collections import OrderedDict
from datetime import date
from importlib import reload

import numpy as np
import pandas as pd
import torch
from path import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.common import utils as c_utils
from src.data import dataset_util
from src.data import movie_sentimement_dataset
from src.features import vocab_utils
from src.models import custom_scheduler
from src.models import senti_transformer, senti_bilstm

reload(dataset_util)
reload(vocab_utils)
reload(movie_sentimement_dataset)

ds_typ = dataset_util.DSType
SPECIALS = ['<unk>', '<pad>', '<sos>', '<eos>']
SYMBOL_TO_IDX = OrderedDict(zip(SPECIALS, range(len(SPECIALS))))

VOCAB_REFRESH = True

NHEAD = 2
N_FOLD = 5
EMBED_SIZE = 200
HIDDEN_SIZE = 160
NLAYER = 3
BATCH_SIZE = 128
CLIP_GRAD_NORM = 0.25
LR = 0.01
NUM_EPOCH = 8
LOG_INTERVAL = 100
PATIENCE = 2
LR_DECAY = .5
model_dir = Path('models')
MODEL_TYPE = 'Bi-LSTM'
MODEL_NAME = f'{MODEL_TYPE}_EP-{NUM_EPOCH}_EM-{EMBED_SIZE}_HS-{HIDDEN_SIZE}_NL-{NLAYER}_NH-{NHEAD}_BS-{BATCH_SIZE}'
MODEL_SAVE_PATH = model_dir.joinpath(f'{MODEL_NAME}.pth')


def phrase_preprocess(x: str) -> str:
    """
    Preprocess phrases for vocab and other transformation

    Parameters
    ----------
    x

    Returns
    -------
    Remove non alphabet characters and lowers the upper case.

    """
    x = x.lower()
    x = re.sub('[^a-zA-z0-9\s]', '', x)  # Remove non alphanumeric
    x = x.strip()
    return x


def tokenizer(x: str):
    # tokenizer = get_tokenizer('basic_english')
    return x.split()


def add_sos_eos(x):
    """
    Add start and end of sentece
    Parameters
    ----------
    x

    Returns
    -------

    """
    return [SYMBOL_TO_IDX['<sos>']] + x + [SYMBOL_TO_IDX['<eos>']]


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
            ypred.append(c_utils.get_ypred(output))
            y.append(targets)
        return {"Loss": total_loss / (len(data_source) - 1),
                "Accuracy": c_utils.multi_acc(torch.cat(y), torch.cat(ypred))}


def submit_df(data_source, model):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    with torch.no_grad():
        ypred = []
        idx = []
        for i, data, targets, src_len in data_source:
            output = model(data, src_len)
            ypred.append(c_utils.get_ypred(output))
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
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()
        if optimizer.state_dict()['state'][0]['exp_avg'].isnan().sum().item() > 0:
            print("optimizer lol")

        total_loss += loss.item()
        if batch % LOG_INTERVAL == 0 and batch > 0:
            cur_loss = total_loss / LOG_INTERVAL
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | train_acc {:8.2f}'.format(
                epoch, batch, len(train_loader), new_lr,
                elapsed * 1000 / LOG_INTERVAL, cur_loss, c_utils.multi_acc(targets, c_utils.get_ypred(output))
            )
            )
            epoch_loss += total_loss
            total_loss = 0
            start_time = time.time()
    return epoch_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device_tensor(x):
    return torch.tensor(x, device=device)


movie_senti_ds = dataset_util.DataSpace(
    train_filename='data/raw/train.tsv.zip',
    test_filename='data/raw/test.tsv.zip',
    get_x=lambda df: df['Phrase'],
    get_y=lambda df: df['Sentiment'] if 'Sentiment' in df else torch.nan,
    sep='\t', compression='zip'
)
vocab_tokenizer = c_utils.sequential_transforms(phrase_preprocess, tokenizer)
train_val_X, train_val_y = movie_senti_ds.get_XY_split(ds_typ.TRAINVAL)
NLABEL = train_val_y.nunique()
wt_y = compute_class_weight("balanced", classes=sorted(np.unique(train_val_y)), y=train_val_y)
if VOCAB_REFRESH:
    vocab = vocab_utils.VocabFactory(specials=SPECIALS, tokenizer=vocab_tokenizer)
    vocab.build_vocab(d_iter=movie_senti_ds.yield_XY_split(ds_typ.TRAINVAL))
else:
    vocab = vocab_utils.VocabFactory.load_vocab(model_dir.joinpath('vocab_obj.pth'), SPECIALS)

NTOKENS = vocab.get_vocab_size()

phrase_transformer = c_utils.sequential_transforms(vocab.transform, add_sos_eos, to_device_tensor)

target_transformer = lambda x: to_device_tensor(x)

batch_transformer = lambda x: pad_sequence(x, padding_value=SYMBOL_TO_IDX['<pad>'])

collate_fn = lambda x: movie_sentimement_dataset.collate_batch(x, batch_transformer)

best_val_loss = None
patience = 0

movie_senti_ds.set_transformer(phrase_transformer, target_transformer)

# movie_senti_ds.gen_validation_index(val_size=.2, stratify=train_val_y, shuffle=True)

test_iter = movie_sentimement_dataset.MovieSentimentDataset(*movie_senti_ds.get_XY_split(ds_typ.TEST), is_test=True)
test_loader = DataLoader(test_iter, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

cv = StratifiedKFold(N_FOLD, shuffle=True)
foldacc = []
for fold, (train_ix, val_ix) in enumerate(cv.split(train_val_X, train_val_y), start=1):
    best_fold_val_acc = 0
    print(model_dir.joinpath(f'{MODEL_NAME}_FOLD_{fold}.pth'))
    # Set train and val index
    movie_senti_ds.set_validation_index(val_ix)
    train_iter = movie_sentimement_dataset.MovieSentimentDataset(*movie_senti_ds.get_XY_split(ds_typ.TRAIN),
                                                                 is_test=False)
    val_iter = movie_sentimement_dataset.MovieSentimentDataset(*movie_senti_ds.get_XY_split(ds_typ.VAL),
                                                               is_test=False)
    train_loader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True
                              )
    val_loader = DataLoader(val_iter, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Reset model
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

    scheduler = custom_scheduler.CustomSchedulerLRPlateau(optimizer,
                                                          "max",
                                                          patience=PATIENCE,
                                                          factor=LR_DECAY,
                                                          verbose=True)

    # Start Epoch

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
            scheduler.step(
                lambda x: c_utils.save_checkpoint(model_dir.joinpath(f'{MODEL_NAME}_FOLD_{fold}.pth'), model, optimizer,
                                                  scheduler, epoch),
                val_acc)
            best_fold_val_acc = max(val_acc, best_fold_val_acc)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    foldacc.append(best_fold_val_acc)

# Run on test data.
res = []

print('=' * 89)
for fold in range(1, N_FOLD + 1):
    model_path = model_dir.joinpath(f'{MODEL_NAME}_FOLD_{fold}.pth')
    with torch.no_grad():
        with open(model_path, 'rb') as f:
            checkpoint = torch.load(f)
            model = checkpoint['model']
            res.append(submit_df(test_loader, model))

res_df = pd.concat([pd.Series(r['pred'], r['index']).rename(f'pred_{i}') for i, r in enumerate(res)], 1)
# Best accuracy from logs
# foldacc = torch.tensor([0.6183, 0.61786, 0.6170, 0.61971])
foldacc = torch.tensor(foldacc)
foldacc_wt = (foldacc / foldacc.sum()).numpy()
preds = res_df.apply(lambda x: (x * foldacc_wt).sum().round(), 1).astype(int)
test_df = movie_senti_ds.get_df_split(ds_typ.TEST)
test_df.loc[res_df.index, 'Sentiment'] = preds
assert test_df['Sentiment'].isnull().sum() == 0
test_df[['PhraseId', 'Sentiment']].to_csv(f'data/submission/submit_{MODEL_NAME}_{date.today()}.csv', index=False)

print(f"Submission file: data/submission/submit_{MODEL_NAME}_{date.today()}.csv")
print('=' * 89)
