import torch
from torch import nn
from torch.nn import functional as F


class LSTM(nn.Module):
    def __init__(self, embedding_dim=64, n_lstms=3, drop_rate=0.1, bidirectional=True):
        super().__init__()
        self.layers = nn.GRU(embedding_dim,
                             embedding_dim,
                             n_lstms,
                             dropout=drop_rate,
                             bidirectional=bidirectional,
                             batch_first=True)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        b, e, l = x.size()
        h_0 = torch.zeros((b, self.embedding_dim, l))
        return self.layers(x, h_0)[0]
