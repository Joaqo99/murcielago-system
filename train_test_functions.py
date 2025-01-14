import torch
import torchaudio
from torch.utils.data import DataLoader
from custom_dataset import GunShotsNoisesDataset, split_dataset
import numpy as np


def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device="cpu"):
    """Performs a training with model trying to learn on data_loader"""
    train_loss, train_acc = 0, 0

    model.to(device)

    model.train()
    for X, y in data_loader:    
        X, y = X.to(device), y.to(device)
        y_logits = model(X).squeeze()

        loss = loss_fn(y_logits, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_logits=y_logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.5f}%")


def test_step(model, data_loader, loss_fn, accuracy_fn, device="cpu"):
    """Performs a training with model trying to learn on data_loader"""
    test_loss, test_acc = 0, 0

    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            test_pred = model(X).squeeze()
            
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_logits=test_pred)

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.5f}%")

def eval_model(model: torch.nn.Module, data_loader, loss_fn, accuracy_fn):
    """
    Returns a dictionary containing the results of model predicting on data_loader
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X)

            # accumulate the loss and acc values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        loss /= len(data_loader)
        acc /= len(data_loader)


    return {"model_name": model.__class__.__name__, "loss": loss.item(), "accuracy": acc}


def accuracy_fn(y_true, y_logits):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    y_pred =  torch.round(torch.sigmoid(y_logits))
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc