"""
training utilities
"""

import os
import pathlib
import time

import torch
from torch import nn


def loss_fn(y_hat, y):
    raise NotImplementedError


def accuracy_fn(y_hat, y):
    raise NotImplementedError


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    accuracy_fn: torch.nn.Module,
    device: torch.device=None
):
    # train mode
    model.train()

    # loss and accuracy
    train_loss = 0.
    train_acc = 0.

    # iterate over given dataloader
    for batch, (X, y) in enumerate(dataloader):
        # put data on target device
        X, y = X.to(device), y.to(device)

        # forward pass
        X_, mu_, sigma_ = model(X)

        # loss & accuracy computation
        loss = loss_fn(X_, X, mu_, sigma_)
        train_loss += loss.item()
        # acc = accuracy_fn(y_hat, y)
        # train_acc += acc.item()

        # zero the parameter gradients
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update the parameters
        optimizer.step()

        # if batch % 32 == 0:
        #     log = f">>> train_step -> batch: {batch}/{len(dataloader)}, " \
        #             f"train_loss: {loss.item() / len(X):.4f}"   # division by batch size looks similar to whats reported in reference
        #     print(log)

    # compute the loss and accuracy for this epoch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn: torch.nn.Module,
    device: torch.device=None
):
    # eval mode
    model.eval()

    # loss and accuracy
    test_loss = 0.
    test_acc = 0.

    # inference mode
    with torch.inference_mode():
        # iterate over given dataloader
        for batch, (X, y) in enumerate(dataloader):
            # put data on target device
            X, y = X.to(device), y.to(device)

            # forward pass
            X_, mu_, sigma_ = model(X)

            # loss & accuracy computation
            loss = loss_fn(X_, X, mu_, sigma_)
            test_loss += loss.item()
            # acc = accuracy_fn(y_hat, y)
            # test_acc += acc.item()

            # if batch % 32 == 0:
            #     log = f">>> test_step -> batch: {batch}/{len(dataloader)}, " \
            #             f"test_loss: {loss.item() / len(X):.4f}"
            #     print(log)

    # compute the loss and accuracy for this epoch
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    accuracy_fn: torch.nn.Module,
    epochs: int=5,
    device: torch.device=None,
):
    """
    performs a training loop on model going over train_dataloader
    and evaluate the model on test_dataloader
    """

    # send the model to target device
    model = model.to(device)

    # define a dictionary to keep track of train and test metrics
    hist_dict = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # start training loop
    t0 = time.perf_counter()
    for epoch in range(1, epochs+1):
        # train step
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device
        )

        # test step
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device
        )

        # append train and test loss
        hist_dict["epoch"].append(epoch)
        hist_dict["train_loss"].append(train_loss)
        hist_dict["train_acc"].append(train_acc)
        hist_dict["test_loss"].append(test_loss)
        hist_dict["test_acc"].append(test_acc)

        # print the progress
        t1 = time.perf_counter()
        elps = t1 - t0
        log = f"epoch: {epoch}/{epochs}, " \
                f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.2f}, " \
                f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.2f}, " \
                f"elapsed: {elps:.2f} sec"
        print(log)

    return hist_dict


def save_model(
    model: torch.nn.Module,
    save_dir: str,
    model_name: str,
):
    save_dir = pathlib.Path(save_dir)
    if save_dir.exists():
        print(f">>> save_dir: {save_dir} already exists")
    else:
        print(f">>> save_dir: {save_dir} does not exist, creating...")
        save_dir.mkdir(parents=True, exist_ok=True)

    if model_name.endswith(".pt") or model_name.endswith(".pth"):
        pass
    else:
        model_name += ".pt"

    save_path = save_dir / model_name
    print(f">>> saving model to: {save_path}")
    torch.save(model.state_dict(), save_path)
