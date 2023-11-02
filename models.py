"""
models
"""

import torch
from torch import nn

from layers import *

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=.5, bias=True):
        super().__init__()
        self.gc1 = GraphConvolutionLayer(in_features, hidden_features, bias=bias)
        self.gc2 = GraphConvolutionLayer(hidden_features, out_features, bias=bias)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = nn.ReLU()(x)
        x = nn.Dropout(self.dropout)(x)
        x = self.gc2(x, adj)
        x = nn.LogSoftmax(dim=1)(x)
        return x
