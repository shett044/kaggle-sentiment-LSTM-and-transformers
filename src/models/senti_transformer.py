import math

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SentimentTransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, nlabel, dropout=0.3):
        super(SentimentTransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.ninp = ninp
        self.nlabel = nlabel
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=1)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Sequential(
            *[nn.Linear(3 * ninp, ninp), nn.Dropout(p=dropout), nn.ReLU(), nn.Linear(ninp, nlabel)])
        # self.decoder_2 = nn.Linear(ninp, nlabel)

    def forward(self, src, pad_mask, has_mask=True):
        self.src_pad_mask = pad_mask
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = Transformer.generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
            self.src_pad_mask = None

        embed_src = self.encoder(src.int()) * math.sqrt(self.ninp)  # [sequence length, batch size, embed dim]
        # Debug
        # for name, params in self.named_parameters():
        #     print("-->name:", name, "-->max_grad:", params.grad.max(), "-->min_grad:", params.grad.min())

        if embed_src.isnan().all():
            for name, params in self.named_parameters():
                print("-->name:", name, "--> grad:", params.grad)

        embed_src = self.pos_encoder(embed_src)  # [sequence length, batch size, embed dim]
        enc_output = self.transformer_encoder(embed_src,
                                              src_key_padding_mask=self.src_pad_mask.bool())  # [sequence length, batch size, embed dim]
        comb_seq_output = torch.cat(
            (enc_output.max(dim=0).values, enc_output.min(dim=0).values, enc_output.mean(dim=0)),
            dim=1)  # [batch_size, 3*embed_dim]

        output = self.decoder(comb_seq_output)  # [batch_size, nlab]
        return output
