import torch
from torch import nn
from torch.nn import functional as F


class LSTM(nn.Module):
    def __init__(self, embedding_dim=64, n_lstms=3, drop_rate=0.1, bidirectional=False):
        super().__init__()
        self.layers = nn.GRU(embedding_dim,
                             embedding_dim,
                             n_lstms,
                             dropout=drop_rate,
                             bidirectional=bidirectional,
                             batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.layers(x)[0].permute(0, 2, 1)