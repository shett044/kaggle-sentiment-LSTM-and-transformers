import re

import torch
from torch.utils.data import Dataset


class MovieSentimentDataset(Dataset):
    def __init__(self, X_df, y_df, is_test: bool):
        self.X_df = X_df.str.lower().str.strip().apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
        self.y_df = y_df
        self.is_test = is_test

    def __len__(self):
        return len(self.X_df)

    def __getitem__(self, idx):
        if self.is_test:
            return idx, self.X_df.iloc[idx], torch.nan
        return idx, self.X_df.iloc[idx], self.y_df.iloc[idx]

    def get_sample_weight(self, class_wt):
        print(f"Using {class_wt=}")
        y_wt = torch.ones(self.y_df.shape[0]).numpy()
        if class_wt is not None:
            y_wt = class_wt[self.y_df]
        return y_wt


def collate_batch(batch, sentence_transformer, batch_transformer, device='cpu'):
    X, y, Xlen = [], [], []
    idx = []
    for _, _X, _y in batch:
        tok_X = sentence_transformer(_X)
        X.append(tok_X)
        Xlen.append(len(tok_X))
        y.append(_y)
        idx.append(_)
    X = batch_transformer(X)
    y = torch.tensor(y, device=device)
    Xlen = torch.tensor(Xlen, device=device)
    seqs_mask = torch.ones(len(batch), Xlen.max())
    for i, x in enumerate(Xlen):
        seqs_mask[i, :x] = 0
    return torch.tensor(idx), X, y, seqs_mask
