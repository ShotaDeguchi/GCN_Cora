"""
layers
"""

import math
import torch
from torch import nn

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # trainable parameters
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input, adj):
        # matrix multiplication between input and weight
        support = torch.mm(input, self.weight)

        # sparse matrix multiplication between adjacency matrix and support
        output = torch.spmm(adj, support)

        # add bias if True
        if self.bias is not None:
            return output + self.bias
        else:
            return output
