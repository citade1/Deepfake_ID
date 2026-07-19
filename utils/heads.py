"""Shared MLP probe: train a small head on frozen features, report AUC."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def fit_head(Xtr, ytr, Xva, yva, seed=0, epochs=20, patience=4):
    torch.manual_seed(seed)
    mlp = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(),
                        nn.Dropout(0.3), nn.Linear(128, 2))
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    best, best_state, stale = -1.0, None, 0
    for _ in range(epochs):
        mlp.train()
        for idx in torch.randperm(len(ytr)).split(64):
            opt.zero_grad()
            F.cross_entropy(mlp(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        if auc(mlp, Xva, yva) > best:
            best, best_state, stale = auc(mlp, Xva, yva), copy.deepcopy(mlp.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    mlp.load_state_dict(best_state)
    return mlp.eval()


@torch.no_grad()
def prob_fake(mlp, X):
    return F.softmax(mlp(X), dim=1)[:, 1]


def auc(mlp, X, y):
    try:
        return roc_auc_score(y, prob_fake(mlp, X))
    except ValueError:
        return float("nan")
