import torch.nn as nn
from . import TE_module
from itertools import combinations
import random


class MDE_Network(nn.Module):
    """
    Multidimensional equivariant subnetwork
    """

    def __init__(self, d_input, d_output, n_layer, d_hidden, dim_list):
        super(MDE_Network, self).__init__()
        layers = []
        self.pre = nn.Linear(d_input, d_hidden)
        for i in range(n_layer):
            layers.append(
                TE_module.MDE_Module_LowFLOPs(in_features=d_hidden, out_features=d_hidden, dim=dim_list[i]))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(d_hidden))
        self.layers = nn.ModuleList(layers)
        self.final = nn.Linear(d_hidden, d_output)

    def forward(self, x):
        """
        :param x: [bs, dim_1, ..., dim_n, d_input]  dim_1, ..., dim_n are dims exhibit multidimensional equivariance
        :return: [bs, dim_1, ..., dim_n, d_output]
        """
        x = self.pre(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.final(x)
        return x


def generate_combinations(n):
    """
    Generate all 2^n-1 subsets (except for the empty set) of [1,...,n].
    For example, when n=3, generate [[1], [2], [3], [1,2], [1,3], [2,3], [1, 2, 3]]
    """
    result = []
    for r in range(1, n + 1):
        result.extend([list(comb) for comb in combinations(range(1, n + 1), r)])
    return result


def generate_pattern1(n):
    """
    Generate [[1], [2], ..., [n]] list
    For example, when n=3, generate [[1], [2], [3]]
    """
    result = [[i] for i in range(1, n + 1)]
    return result


def generate_patterns(n_layer, n_dim, pattern='original', dim_per_layer=2):
    """
    Generate the dim_list of different patterns
    """
    if pattern == 'original':
        result = [generate_combinations(n_dim) for _ in range(n_layer)]

    elif pattern == 'pattern1':
        result = [generate_pattern1(n_dim) for _ in range(n_layer)]

    elif pattern == 'pattern2':
        if n_layer * dim_per_layer < n_dim:
            raise ValueError("The number of layers and dimensions of the MDE network do not match.")
        if dim_per_layer > n_dim:
            raise ValueError("The dim_per_layer is too large.")
        sequence = list(range(1, n_dim + 1))
        while True:
            result_sequence = []
            for _ in range(n_layer):
                result_sequence = result_sequence + random.sample(sequence, dim_per_layer)
            if all(item in result_sequence for item in sequence):
                break
        result = [[[result_sequence[i+j]] for j in range(dim_per_layer)] for i in range(0, len(result_sequence), dim_per_layer)]

    elif pattern == 'pattern3':
        if n_layer < n_dim:
            raise ValueError("The number of layers and dimensions of the MDE network do not match.")
        sequence = list(range(1, n_dim + 1))
        while True:
            result_sequence = []
            for _ in range(n_layer):
                result_sequence = result_sequence + random.sample(sequence, 1)
            if all(item in result_sequence for item in sequence):
                break
        result = [[[result_sequence[i]]] for i in range(n_layer)]

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return result
