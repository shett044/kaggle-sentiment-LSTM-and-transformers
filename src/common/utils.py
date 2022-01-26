import torch
from sklearn.metrics import accuracy_score


def sequential_transforms(*transforms):
    """
    Perform sequential tranfromation
    Parameters
    ----------
    transforms

    Returns
    -------

    """

    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def get_ypred(ypred: torch.Tensor) -> torch.Tensor:
    """
    Perform softmax and get max prediction

    Parameters
    ----------
    ypred

    Returns
    -------
    Tensor with max prediction
    """
    with torch.no_grad():
        y_pred_softmax = torch.log_softmax(ypred, dim=-1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=-1)
    return y_pred_tags


def multi_acc(y_test: torch.Tensor, y_pred: torch.Tensor, class_weight=None) -> float:
    if class_weight is None:
        class_weight_sample = torch.ones_like(y_test)
    else:
        class_weight_sample = class_weight[y_test]

    acc = accuracy_score(y_test, y_pred, sample_weight=class_weight_sample)
    return acc


def save_checkpoint(model_path: str, model, optimizer=None, scheduler=None, epoch=0):
    with open(model_path, 'wb') as f:
        print(f"Saving the best model : {model_path}")
        torch.save({
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": epoch
        }, f)
