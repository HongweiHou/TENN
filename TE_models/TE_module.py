import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math


class MDI_Module(nn.Module):
    """
    Multidimensional Invariant Module.
    Applies permutation-invariant operations (e.g., pooling and attention) along specified axes.

    Args:
        d_feature (int): Feature dimension.
        num_heads (int): Number of attention heads.
        dim (list): List of axes indices for invariance.
    """

    def __init__(self, d_feature, num_heads, dim):
        super(MDI_Module, self).__init__()
        self.dim = dim
        layers = []
        for d in dim:
            layers.append(MDI_Layer(d_feature, num_heads, d))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Args:
            x: [batch_size, M1, M2, ..., MN, d_feature]
        Returns:
            Tensor with specified invariant axes pooled out.
        """
        sp = x.size()
        sp = list(sp)
        # Perform invariant modules on all dimensions that exhibit invariance.
        for i, layer in enumerate(self.layers):
            x = layer(x)
        dim = sorted(self.dim, reverse=True)
        # Remove the indices of dimensions that have been processed.
        for d in dim:
            del sp[d]
        x = torch.reshape(x, sp)
        return x


class MDI_Layer(nn.Module):
    """
    Permutation invariant layer
    """

    def __init__(self, d_feature, num_heads, dim):
        super(MDI_Layer, self).__init__()
        self.s = nn.Parameter(torch.Tensor(1, 1, d_feature))
        nn.init.xavier_uniform_(self.s)

        self.d_feature = d_feature
        self.num_heads = num_heads
        self.fc_q = nn.Linear(d_feature, d_feature)
        self.fc_k = nn.Linear(d_feature, d_feature)
        self.fc_v = nn.Linear(d_feature, d_feature)
        self.fc_o = nn.Linear(d_feature, d_feature)

        self.dim = dim

    def forward(self, x):
        """
        :param x: [bs, ..., dim_to_pool, ... , d_feature]
        :return: [bs, ..., 1 , ..., d_feature]
        """
        x = torch.moveaxis(x, self.dim, -2)
        sp = x.size()
        x = x.flatten(start_dim=0, end_dim=-3)

        q = self.fc_q(self.s.repeat(x.size(0), 1, 1))
        k, v = self.fc_k(x), self.fc_v(x)
        dim_split = self.d_feature // self.num_heads
        q_ = torch.cat(q.split(dim_split, 2), 0)
        k_ = torch.cat(k.split(dim_split, 2), 0)
        v_ = torch.cat(v.split(dim_split, 2), 0)

        a = torch.softmax(q_.bmm(k_.transpose(1, 2)) / math.sqrt(self.d_feature), 2)
        o = torch.cat((q_ + a.bmm(v_)).split(q.size(0), 0), 2)
        o = o + F.relu(self.fc_o(o))

        sp = list(sp)
        sp[-2] = 1
        x = torch.reshape(o, sp)
        x = torch.moveaxis(x, -2, self.dim)
        return x


class HOE_1_2_Module(nn.Module):
    """
     1-2 order equivariant module.
    """

    def __init__(self, d_input, d_output):
        super(HOE_1_2_Module, self).__init__()
        self.fc1 = nn.Linear(5 * d_input, d_input)
        self.fc2 = nn.Linear(d_input, d_output)
        self.ln = nn.LayerNorm(d_input)
        self.act = nn.ReLU()
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1, d_input))

    def forward(self, x):
        """
        :param x: [bs, ue_num, rx_ant_num, d_hidden]
        :return: [bs, ue_num, rx_ant_num, rx_ant_num, d_hidden]
        """
        # [bs, ue_num, rx_ant_num, d_hidden]
        sp = x.size()
        # [bs, ue_num, d_hidden, rx_ant_num]
        x = x.permute(0, 1, 3, 2)
        # [bs, ue_num, d_hidden, rx_ant_num, rx_ant_num]
        x1 = torch.diag_embed(x)
        # [bs, ue_num, d_hidden, rx_ant_num, rx_ant_num]
        x2 = x.unsqueeze(-1).repeat(1, 1, 1, 1, sp[2])
        # [bs, ue_num, d_hidden, rx_ant_num, rx_ant_num]
        x3 = x.unsqueeze(-2).repeat(1, 1, 1, sp[2], 1)
        # [bs, ue_num, d_hidden, rx_ant_num, rx_ant_num]
        x4 = torch.diag_embed(torch.mean(x, dim=3, keepdim=True).expand_as(x))
        # [bs, ue_num, d_hidden, rx_ant_num, rx_ant_num]
        x5 = torch.mean(x, dim=3, keepdim=True).expand_as(x).unsqueeze(-1).repeat(1, 1, 1, 1, sp[2])
        xAll = torch.cat([x1, x2, x3, x4, x5], 2).permute(0, 1, 3, 4, 2)
        # [bs, ue_num, rx_ant_num, rx_ant_num, d_hidden]
        y = self.act(self.ln(self.fc1(xAll)))

        bias_rep = self.bias.repeat(sp[0], sp[1], sp[2], sp[2], 1)
        device = bias_rep.device
        # [1, 1, rx_ant_num, rx_ant_num, 1]
        mask = torch.eye(sp[2]).unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(device)
        bias_rep = bias_rep * mask
        y = y + bias_rep

        y = self.fc2(y)
        return y


class MDE_Module(nn.Module):
    """
    Multidimensional equivariant module
    """

    def __init__(self, in_features, out_features, dim=[1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]):
        super(MDE_Module, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(in_features * (1 + len(dim)), out_features)

    def forward(self, x):
        """
        :param x: [bs, ue_num, rx_ant_num, tx_ant_num, in_features]
        :return: [bs, ue_num, rx_ant_num, tx_ant_num, out_features]
        """
        pooled = [torch.mean(x, d, keepdim=True).expand_as(x) for d in self.dim]
        state = torch.cat([x] + pooled, dim=-1)
        y = self.linear(state)
        return y


class MDE_Module_LowFLOPs(nn.Module):
    """
    Multidimensional equivariant module
    Perform linear processing before applying expansion, and then sum up.
    Reduction of FLOPs is achieved by avoiding redundant operations.
    """

    def __init__(self, in_features, out_features, dim=[1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]):
        super(MDE_Module_LowFLOPs, self).__init__()
        self.dim = dim
        layers = []
        self.linear = nn.Linear(in_features, out_features)
        for d in dim:
            layers.append(nn.Linear(in_features, out_features, bias=False))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        :param x: [bs, ue_num, rx_ant_num, tx_ant_num, in_features]
        :return: [bs, ue_num, rx_ant_num, tx_ant_num, out_features]
        """
        y = self.linear(x)
        for i, layer in enumerate(self.layers):
            y = y + layer(torch.mean(x, self.dim[i], keepdim=True)).expand_as(y)
        return y
