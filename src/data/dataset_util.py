from typing import Iterable, Callable

import pandas as pd
from sklearn.model_selection import train_test_split


class DSType:
    """Dataset type"""
    ALL = "ALL"
    TRAIN = "TRAIN"
    VAL = "VAL"
    TRAINVAL = "TRAINVAL"
    TEST = "TEST"


class DataTypeNotCreated(Exception):
    """Custom error that is raised when not enough vacation days are available."""

    def __init__(self, message: str = "DataType not present/created") -> None:
        self.message = message
        super().__init__(message)


class DFType:
    def __init__(self, trainval_df: pd.DataFrame = None, test_df: pd.DataFrame = None,
                 train_df: pd.DataFrame = None) -> None:
        self.TRAINVAL = trainval_df
        self.TEST = test_df
        self.ALL = self.TRAIN = self.VAL = None

        if trainval_df is not None and test_df is not None:
            self.ALL = pd.concat([trainval_df, test_df])

        if train_df is not None:
            self.TRAIN = train_df
            self.VAL = self.TRAINVAL.loc[self.TRAINVAL.index.difference(self.TRAIN.index)]

    def set_validation_index(self, val_idx):
        self.VAL = self.TRAINVAL.loc[val_idx]
        self.TRAIN = self.TRAINVAL.loc[self.TRAINVAL.index.difference(self.VAL.index)]

    def set_train_index(self, train_idx):
        self.TRAIN = self.TRAINVAL.loc[train_idx]
        self.VAL = self.TRAINVAL.loc[self.TRAINVAL.index.difference(self.TRAIN.index)]

    def get_df_by_DSType(self, type: DSType):
        if type == DSType.ALL and self.ALL is not None:
            return self.ALL
        elif type == DSType.TRAINVAL and self.TRAINVAL is not None:
            return self.TRAINVAL
        elif type == DSType.TEST and self.TEST is not None:
            return self.TEST
        elif type == DSType.TRAIN and self.TRAIN is not None:
            return self.TRAIN
        elif type == DSType.VAL and self.VAL is not None:
            return self.VAL
        else:
            raise DataTypeNotCreated()


class DataSpace:
    def __init__(self, train_filename, test_filename, get_x: Callable = None, get_y: Callable = None, **pdcsv_kw):
        self.dftype = DFType(trainval_df=pd.read_csv(train_filename, **pdcsv_kw),
                             test_df=pd.read_csv(test_filename, **pdcsv_kw))
        self.getX = get_x or (lambda x: x)
        self.getY = get_y or (lambda x: x)
        self.X_transformer = lambda x: x
        self.y_transformer = lambda y: y

    def set_transformer(self, x_trx: Callable = None, y_trx: Callable = None):
        if x_trx:
            self.X_transformer = x_trx
        if y_trx:
            self.y_transformer = y_trx

    def gen_validation_index(self, val_size, stratify=None, shuffle=None):
        train_idx, val_idx = train_test_split(self.dftype.TRAINVAL.index, test_size=val_size, stratify=stratify,
                                              shuffle=shuffle)
        self.set_validation_index(val_idx)
        return train_idx, val_idx

    def set_validation_index(self, val_index=None):
        self.dftype.set_validation_index(val_index)

    def set_validation_index(self, val_index):
        self.dftype.set_validation_index(val_index)

    def setXY_callable(self, get_x: Callable, get_y: Callable):
        self.getX = get_x
        self.getY = get_y

    def get_df_split(self, dataset_type: DSType) -> pd.DataFrame:
        return self.dftype.get_df_by_DSType(dataset_type)

    def get_XY_split(self, dataset_type: DSType, use_transformer=True) -> pd.DataFrame:
        df = self.get_df_split(dataset_type)
        X, y = self.getX(df), self.getY(df)
        if use_transformer:
            X = X.apply(self.X_transformer)
            if dataset_type != 'TEST':
                y = y.apply(self.y_transformer)
        return X, y

    def yield_XY_split(self, dataset_type: DSType, use_transformer=True) -> Iterable:
        for i, r in self.get_df_split(dataset_type).iterrows():
            X, y = self.getX(r), self.getY(r)
            if use_transformer:
                X, y = self.X_transformer(X), self.y_transformer(y)
            yield X, y

    def yield_X_split(self, dataset_type: DSType, use_transformer=True) -> Iterable:
        for i, r in self.get_df_split(dataset_type).iterrows():
            X= self.getX(r)
            if use_transformer:
                X = self.X_transformer(X)
            yield X

    def yield_y_split(self, dataset_type: DSType, use_transformer=True) -> Iterable:
        for i, r in self.get_df_split(dataset_type).iterrows():
            y = self.getY(r)
            if use_transformer:
                y = self.y_transformer(y)
            yield y
