from typing import Callable, Any, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class MovieSentimentDataset(Dataset):
    def __init__(self, X_df, y_df, is_test: bool):
        self.X_df = X_df
        # .apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
        self.y_df = y_df
        self.is_test = is_test

    def __len__(self):
        return len(self.X_df)

    def __getitem__(self, idx):
        if self.is_test:
            return idx, self.X_df.iloc[idx], torch.tensor(torch.nan)
        return idx, self.X_df.iloc[idx], self.y_df.iloc[idx]

    def get_sample_weight(self, class_wt):
        print(f"Using {class_wt=}")
        y_wt = torch.ones(self.y_df.shape[0]).numpy()
        if class_wt is not None:
            y_wt = class_wt[self.y_df]
        return y_wt


def collate_batch(batch: Any, batch_transformer: Callable) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Extra processing with the batches. Generates sequence mask

    Parameters
    ----------
    batch: Batch containing index, X, and Y
    batch_transformer : Adds padding to the X

    Returns
    -------
    Index, X, y and sequence mask (True for padding)
    -------

    """
    X, y, Xlen = [], [], []
    idx = []
    for _, tok_X, _y in batch:
        X.append(tok_X)
        Xlen.append(len(tok_X))
        y.append(_y)
        idx.append(_)
    X = batch_transformer(X)
    y = torch.stack(y)
    Xlen = torch.tensor(Xlen, device=y.device)
    seqs_mask = torch.ones(len(batch), Xlen.max())

    for i, x in enumerate(Xlen):
        seqs_mask[i, :x] = 0
    return torch.tensor(idx), X, y, seqs_mask
