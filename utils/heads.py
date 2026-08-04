"""Shared MLP probe: train a small head on frozen features, report AUC."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def fit_head(Xtr, ytr, Xva, yva, seed=0, epochs=20, patience=4):
    # init fixed by `seed` on purpose: run-to-run variance comes from the split, not the weights
    torch.manual_seed(seed)
    mlp = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(),
                        nn.Dropout(0.3), nn.Linear(128, 2))
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    best, best_state, stale = -1.0, copy.deepcopy(mlp.state_dict()), 0   # fall back to init if val AUC is always nan
    for _ in range(epochs):
        mlp.train()
        for idx in torch.randperm(len(ytr)).split(64):
            opt.zero_grad()
            F.cross_entropy(mlp(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        mlp.eval()                                           # no dropout while validating
        va = auc(mlp, Xva, yva)
        if va > best:
            best, best_state, stale = va, copy.deepcopy(mlp.state_dict()), 0
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
        p = prob_fake(mlp, X).cpu().numpy()                 # sklearn needs numpy, not a (possibly non-CPU) tensor
        y = y.cpu().numpy() if torch.is_tensor(y) else y
        return roc_auc_score(y, p)
    except ValueError:
        return float("nan")
