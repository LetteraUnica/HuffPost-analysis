import torch
from torch import nn
from torch.nn import functional as F

class ResNetBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, drop_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(nn.Dropout(p=drop_rate),
                                    nn.Conv1d(channels, channels, kernel_size, padding="same"),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(channels)
        )

    def forward(self, x):
        return self.layers(x)


class ResNet(nn.Module):
    def __init__(self, channels, num_blocks=2, drop_rate=0.1, kernel_size=3, skip=True):
        super().__init__()
        self.skip = skip
        self.layers = nn.Sequential(*[ResNetBlock(channels, kernel_size, drop_rate) for _ in range(num_blocks)])

    def forward(self, x):
        if self.skip:
            return x + self.layers(x)
        return self.layers(x)


class CNN(nn.Module):
    def __init__(self, embedding_dim=64, n_resnets=3, blocks_per_resnet=2, kernel_size=3, drop_rate=0.1, skip=True):
        super().__init__()
        self.layers = nn.Sequential(*[ResNet(embedding_dim, blocks_per_resnet, drop_rate, kernel_size, skip) for _ in range(n_resnets)])
    
    def forward(self, x):
        return self.layers(x)


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, out_dim, net):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.net = net
        self.classifier = nn.Linear(embedding_dim, out_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.net(x)
        return self.classifier(x.mean(dim=2))


        