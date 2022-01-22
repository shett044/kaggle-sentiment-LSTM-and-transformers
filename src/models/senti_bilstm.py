import sys

import torch
from torch import nn


class SentimentBiLSTM(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """

    def __init__(self, ntoken, embed_size, hidden_size, num_layers=1, output_size=5, dropout_rate=0.2):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(SentimentBiLSTM, self).__init__()
        self.embed_size = embed_size
        self.ntoken = ntoken
        self.model_embeddings = nn.Embedding(ntoken, embed_size, padding_idx=1)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # default values
        self.encoder = None
        # self.decoder = None
        self.h_projection = None
        self.c_projection = None
        self.dropout = None

        self.encoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout_rate,
        )

        # self.decoder = nn.LSTMCell(input_size= embed_size + hidden_size, hidden_size= hidden_size)
        self.h_projection = nn.Linear(
            out_features=hidden_size, in_features=2 * hidden_size, bias=False
        )
        self.bn_hproj = nn.BatchNorm1d(hidden_size)
        self.c_projection = nn.Linear(
            out_features=hidden_size, in_features=2 * hidden_size, bias=False
        )
        self.bn_cproj = nn.BatchNorm1d(hidden_size)
        self.enc_avg_projection = nn.Linear(
            out_features=hidden_size, in_features=2 * hidden_size, bias=False
        )
        self.enc_min_projection = nn.Linear(
            out_features=hidden_size, in_features=2 * hidden_size, bias=False
        )
        self.enc_max_projection = nn.Linear(
            out_features=hidden_size, in_features=2 * hidden_size, bias=False
        )
        self.bn_comb_out_proj = nn.BatchNorm1d(hidden_size)

        self.combined_output_projection = nn.Linear(
            out_features=hidden_size, in_features=5 * hidden_size, bias=False
        )

        self.target_vocab_projection = nn.Linear(
            in_features=hidden_size, out_features=output_size, bias=False
        )
        self.dropout = nn.Dropout(p=dropout_rate)

        ### END YOUR CODE

    def forward(self, source_padded, pad_mask) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """

        # Compute sentence lengths

        source_lengths = (pad_mask == 0).sum(dim=1)

        enc_hiddens = self.encode(source_padded, source_lengths)  # return: batch, 5 * hidden size
        comb_output = self.combined_output_projection(enc_hiddens)
        comb_output = self.bn_comb_out_proj(comb_output)
        combined_outputs = self.dropout(torch.relu(comb_output))
        Y_bar = self.target_vocab_projection(combined_outputs)
        return Y_bar

    def encode(
            self, source_padded: torch.Tensor, src_len
    ):
        enc_hiddens, dec_init_state = None, None
        X = self.model_embeddings(source_padded)

        # Pack the data
        packed_X = nn.utils.rnn.pack_padded_sequence(X, src_len, enforce_sorted=False)

        enc_hiddens, (last_hidden, last_cell) = self.encoder(packed_X)
        enc_hiddens, hidden_seq_lengths = nn.utils.rnn.pad_packed_sequence(enc_hiddens)
        s, b, h = tuple(enc_hiddens.size())  # enc hidden : Sentence length, Batch, hiddensize = 2*h
        # enc_hiddens = torch.permute(enc_hiddens, (1, 0, 2)) # s, b, h  -> b, s, h
        init_decoder_hidden = self.bn_hproj(self.h_projection(
            torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        ))
        init_decoder_cell = self.bn_cproj(self.c_projection(
            torch.cat((last_cell[0], last_cell[1]), dim=1)
        ))
        enc_hiddens_max = self.enc_max_projection(
            enc_hiddens.max(dim=0).values)  # batch, 2* hidden size =>  batch,hidden size
        enc_hiddens_min = self.enc_min_projection(
            enc_hiddens.min(dim=0).values)  # batch, 2* hidden size =>  batch,hidden size
        enc_hiddens_mean = self.enc_avg_projection(
            enc_hiddens.mean(dim=0))  # batch, 2* hidden size =>  batch,hidden size
        enc_hiddens_complete = torch.cat(
            (
                init_decoder_hidden,
                init_decoder_cell,
                enc_hiddens_max,
                enc_hiddens_min,
                enc_hiddens_mean,
            ),
            dim=1,
        )  # batch, 5 * hidden size

        return enc_hiddens_complete

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params["args"]
        model = SentimentBiLSTM(**args)
        model.load_state_dict(params["state_dict"])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print("save model parameters to [%s]" % path, file=sys.stderr)

        params = {
            "args": dict(
                ntoken=self.ntoken,
                embed_size=self.embed_size,
                hidden_size=self.hidden_size,
                dropout_rate=self.dropout_rate,
            ),
            "vocab": self.vocab,
            "state_dict": self.state_dict(),
        }

        torch.save(params, path)
