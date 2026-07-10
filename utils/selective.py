"""Confidence-gated selective LID: the raw-CLS probe decides confident cases; the
raw+LID probe re-scores only the low-confidence ones (|p_raw-0.5| < tau), where
LID's signal concentrates. tau is set on validation to gate a target fraction."""
import torch
import torch.nn.functional as F


class SelectiveLID:
    def __init__(self, raw_head, both_head, tau):
        self.raw, self.both, self.tau = raw_head.eval(), both_head.eval(), tau

    @torch.no_grad()
    def scores(self, raw_x, both_x):
        p = F.softmax(self.raw(raw_x), dim=1)[:, 1].clone()
        unc = (p - 0.5).abs() < self.tau
        if unc.any():
            p[unc] = F.softmax(self.both(both_x[unc]), dim=1)[:, 1]
        return p

    @staticmethod
    def tau_for(p_raw, frac):
        return torch.quantile((p_raw - 0.5).abs(), frac).item()
