"""
train
"""

import os
import time
import pathlib
import argparse
import tarfile
import requests

import torch
from torch import nn
import matplotlib.pyplot as plt

from data import *
from layers import *
from models import *
from train_utils import *

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--l2", type=float, default=5e-4)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--hidden_features", type=int, default=16)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()


def main():
    # set seed
    set_seed(args.seed)

    # plot settings
    plot_settings()

    # main
    _main()


def _main():
    # set seed
    set_seed(args.seed)
    print(f">>> seed: {args.seed}")

    # set device
    device = torch.device(args.device)
    print(f">>> device: {device}")

    # load data
    cora_url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    print(">>> downloading dataset...")
    with requests.get(cora_url, stream=True) as tgz_file:
        with tarfile.open(fileobj=tgz_file.raw, mode="r:gz") as tgz_object:
            tgz_object.extractall()

    print(">>> loading data...")
    features, labels, adj_mat = load_cora(device=device)
    idx = torch.randperm(len(labels)).to(device)
    idx_test, idx_val, idx_train = idx[:1000], idx[1000:1500], idx[1500:]

    # loss & accuracy function
    loss_fn = nn.NLLLoss()
    def acc_fn(logits, labels):
        _, preds = logits.max(dim=1)
        correct = preds.eq(labels).double()
        acc = correct.sum() / len(correct)
        return acc.item()

    ########################################
    # model 0: MLP (adjacency matrix -> identity matrix)
    ########################################
    print(">>> training MLP...")
    model_0 = GCN(
        in_features=features.size(1),
        hidden_features=args.hidden_features,
        out_features=labels.max().item()+1,
        dropout=args.dropout,
        bias=True
    ).to(device)
    optimizer_0 = torch.optim.Adam(
        model_0.parameters(),
        lr=args.lr,
        weight_decay=args.l2
    )

    history_0 = {
        "epoch": [],
        "loss": [],
        "acc": [],
        "loss_val": [],
        "acc_val": [],
        "best_loss": [],
        "wait": [],
        "time": []
    }

    # train
    print("\n>>> training...")
    t0 = time.time()
    best_loss = float("inf")
    wait = 0
    for epoch in range(0, args.epochs+1):
        # train
        model_0.train()
        # forward
        logits = model_0(features, torch.eye(adj_mat.size(0)).to(device))
        loss = loss_fn(logits[idx_train], labels[idx_train])
        acc = acc_fn(logits[idx_train], labels[idx_train])

        # backward
        optimizer_0.zero_grad()
        loss.backward()
        optimizer_0.step()

        # validation
        model_0.eval()
        with torch.inference_mode():
            logits = model_0(features, torch.eye(adj_mat.size(0)).to(device))
            loss_val = loss_fn(logits[idx_val], labels[idx_val])
            acc_val = acc_fn(logits[idx_val], labels[idx_val])

        # early stopping
        if loss_val < best_loss:
            best_loss = loss_val
            wait = 0
        else:
            wait += 1
            if wait == args.patience:
                print(f"early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            t1 = time.time()
            elps = t1 - t0
            print(f"epoch: {epoch:03d}, "
                    f"loss: {loss:.3f}, acc: {acc:.3f}, "
                    f"loss_val: {loss_val:.3f}, acc_val: {acc_val:.3f}, "
                    f"best_loss: {best_loss:.3f}, wait: {wait:02d}, "
                    f"time: {elps:.2f} s")
            t0 = time.time()

        history_0["epoch"].append(epoch)
        history_0["loss"].append(loss.item())
        history_0["acc"].append(acc)
        history_0["loss_val"].append(loss_val.item())
        history_0["acc_val"].append(acc_val)

    # test
    print("\n>>> testing...")
    model_0.eval()
    with torch.inference_mode():
        logits = model_0(features, torch.eye(adj_mat.size(0)).to(device))
        loss_test = loss_fn(logits[idx_test], labels[idx_test])
        acc_test = acc_fn(logits[idx_test], labels[idx_test])
    print(f"loss_test: {loss_test:.3f}, acc_test: {acc_test:.3f}")

    # save
    print("\n>>> saving...")
    save_model(
        model=model_0,
        save_dir="saved_models",
        model_name="mlp.pth"
    )


    ####################
    # model 1: GCN
    ####################
    print(">>> training GCN...")
    model_1 = GCN(
        in_features=features.size(1),
        hidden_features=args.hidden_features,
        out_features=labels.max().item() + 1,
        dropout=args.dropout,
        bias=True
    ).to(device)
    optimizer_1 = torch.optim.Adam(
        model_1.parameters(),
        lr=args.lr,
        weight_decay=args.l2
    )

    history_1 = {
        "epoch": [],
        "loss": [],
        "acc": [],
        "loss_val": [],
        "acc_val": [],
        "best_loss": [],
        "wait": [],
        "time": []
    }

    # train
    print("\n>>> training...")
    t0 = time.time()
    best_loss = float("inf")
    wait = 0
    for epoch in range(0, args.epochs+1):
        # train
        model_1.train()
        # forward
        logits = model_1(features, adj_mat)
        loss = loss_fn(logits[idx_train], labels[idx_train])
        acc = acc_fn(logits[idx_train], labels[idx_train])

        # backward
        optimizer_1.zero_grad()
        loss.backward()
        optimizer_1.step()

        # validation
        model_1.eval()
        with torch.inference_mode():
            logits = model_1(features, adj_mat)
            loss_val = loss_fn(logits[idx_val], labels[idx_val])
            acc_val = acc_fn(logits[idx_val], labels[idx_val])

        # early stopping
        if loss_val < best_loss:
            best_loss = loss_val
            wait = 0
        else:
            wait += 1
            if wait == args.patience:
                print(f"early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            t1 = time.time()
            elps = t1 - t0
            print(f"epoch: {epoch:03d}, "
                    f"loss: {loss:.3f}, acc: {acc:.3f}, "
                    f"loss_val: {loss_val:.3f}, acc_val: {acc_val:.3f}, "
                    f"best_loss: {best_loss:.3f}, wait: {wait:02d}, "
                    f"time: {elps:.2f} s")
            t0 = time.time()

        history_1["epoch"].append(epoch)
        history_1["loss"].append(loss.item())
        history_1["acc"].append(acc)
        history_1["loss_val"].append(loss_val.item())
        history_1["acc_val"].append(acc_val)

    # test
    print("\n>>> testing...")
    model_1.eval()
    with torch.inference_mode():
        logits = model_1(features, adj_mat)
        loss_test = loss_fn(logits[idx_test], labels[idx_test])
        acc_test = acc_fn(logits[idx_test], labels[idx_test])
    print(f"loss_test: {loss_test:.3f}, acc_test: {acc_test:.3f}")

    # save
    print("\n>>> saving...")
    save_model(
        model=model_1,
        save_dir="saved_models",
        model_name="gcn.pth"
    )

    # plot
    path_results = pathlib.Path("results")
    path_results.mkdir(exist_ok=True)
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(history_0["epoch"], history_0["loss"], label="train")
    plt.plot(history_0["epoch"], history_0["loss_val"], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("MLP")
    plt.ylim(.5, 2.)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_1["epoch"], history_1["loss"], label="train")
    plt.plot(history_1["epoch"], history_1["loss_val"], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("GCN")
    plt.ylim(.5, 2.)
    plt.legend()

    plt.tight_layout()
    plt.savefig(path_results / "loss.png")
    plt.close()


    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(history_0["epoch"], history_0["acc"], label="train")
    plt.plot(history_0["epoch"], history_0["acc_val"], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("MLP")
    plt.ylim(.1, .9)
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(history_1["epoch"], history_1["acc"], label="train")
    plt.plot(history_1["epoch"], history_1["acc_val"], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("GCN")
    plt.ylim(.1, .9)
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(path_results / "accuracy.png")
    plt.close()




def set_seed(seed):
    # set seed
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    # if torch.backends.mps.is_available():
    #     torch.mps.manual_seed(seed)


def plot_settings():
    plt.style.use("ggplot")
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    # plt.rcParams["font.size"] = 12
    plt.rcParams["legend.loc"] = "best"
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 1.
    plt.rcParams["savefig.dpi"] = 200


if __name__ == "__main__":
    main()
