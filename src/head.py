"""Stage 5 head (wiki/baseline.md): 4 fps dilated TCN over the stage-4 features.

to4(a)            -> 8 fps rows averaged in pairs (use it on features and on times).
Head(d, mean, std)(x [n,d]) -> frame logits [n]; inputs standardized with stats stored in the head.
                  x [n,C,d] (E1 grid cells): the TCN runs on each cell with shared weights; frame logit = max over cells.
score(logits, t4) -> q = sigmoid(mean of the top-4 logits), t = time of the max logit.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

H, DIL, DROP, TOPK = 128, (1, 2, 4, 8, 16), 0.3, 4


def to4(a):  # [N,...] at 8 fps -> [ceil(N/2),...] at 4 fps; an odd last row stands alone
    a = np.asarray(a, dtype=np.float32)
    if len(a) % 2:
        a = np.concatenate([a, a[-1:]])
    return a.reshape(len(a) // 2, 2, *a.shape[1:]).mean(1)


class Head(nn.Module):
    def __init__(self, d, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.inp = nn.Linear(d, H)
        self.convs = nn.ModuleList(nn.Conv1d(H, H, 3, dilation=k) for k in DIL)
        self.drop = nn.Dropout(DROP)
        self.out = nn.Conv1d(H, 1, 1)

    def forward(self, x):
        h = self.inp((x - self.mean) / self.std)
        h = h.permute(1, 2, 0) if x.dim() == 3 else h.T[None]  # [C, H, n] or [1, H, n]
        for conv, k in zip(self.convs, DIL):
            h = h + self.drop(F.relu(conv(F.pad(h, (k, k), mode="replicate"))))
        return self.out(h)[:, 0].max(0).values if x.dim() == 3 else self.out(h)[0, 0]


def video_logit(logits):
    return logits.topk(min(TOPK, len(logits))).values.mean()


def score(logits, t4):
    logits = torch.as_tensor(logits)
    return float(torch.sigmoid(video_logit(logits))), float(t4[int(logits.argmax())])


def full_score(lg, t):  # {group: L_k}: q = sigmoid(mean_k video_logit(L_k)), t = time of the max of mean_k L_k (plan.md 8)
    v = np.mean([float(video_logit(torch.from_numpy(l))) for l in lg.values()])
    return float(1 / (1 + np.exp(-v))), float(t[int(np.argmax(np.mean(list(lg.values()), 0)))])
