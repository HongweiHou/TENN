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
        # Perform invariant modules on all dimensions that exhibit invariance (in descending order to avoid index shift).
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
    Permutation-Invariant Attention Layer for a single axis.
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
        Args:
            x: [..., dim_to_pool, ..., d_feature]
        Returns:
            Tensor pooled over dim_to_pool axis.
        """
        # Move the axis to be pooled to the second last position
        x = torch.moveaxis(x, self.dim, -2)
        sp = x.size()
        # Flatten all batch and spatial dimensions except the axis to pool and feature
        x = x.flatten(start_dim=0, end_dim=-3)
        # Repeat query vector for batch
        q = self.fc_q(self.s.repeat(x.size(0), 1, 1))
        k, v = self.fc_k(x), self.fc_v(x)
        # Multi-head split
        dim_split = self.d_feature // self.num_heads
        q_ = torch.cat(q.split(dim_split, 2), 0)
        k_ = torch.cat(k.split(dim_split, 2), 0)
        v_ = torch.cat(v.split(dim_split, 2), 0)
        # Attention
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
    1-2 Order Equivariant Module.

    Args:
        d_input (int): Input feature dimension.
        d_output (int): Output feature dimension.
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
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, N, M, F)
        Returns:
            torch.Tensor: output tensor of shape (batch_size, N, M, M, D)
        """
        B, N, M, F = x.shape
        x = x.permute(0, 1, 3, 2) # shape = (B, N, F, M)
        
        # Build the five equivariant components (all shape (B, N, F, M, M))
        x1 = torch.diag_embed(x)
        x2 = x.unsqueeze(-1).repeat(1, 1, 1, 1, M)
        x3 = x.unsqueeze(-2).repeat(1, 1, 1, M, 1)
        x4 = torch.diag_embed(torch.mean(x, dim=3, keepdim=True).expand_as(x))
        x5 = torch.mean(x, dim=3, keepdim=True).expand_as(x).unsqueeze(-1).repeat(1, 1, 1, 1, M)

        # Concatenate and reshape to (B, N, M, M, 5*F)
        xAll = torch.cat([x1, x2, x3, x4, x5], 2).permute(0, 1, 3, 4, 2)
        # Project back to F features, normalize and activate
        y = self.act(self.ln(self.fc1(xAll)))

        # Add learnable bias on the diagonal
        bias_rep = self.bias.repeat(sp[0], sp[1], sp[2], sp[2], 1)
        device = bias_rep.device
        mask = torch.eye(sp[2]).unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(device)
        bias_rep = bias_rep * mask
        y = y + bias_rep

        y = self.fc2(y)
        return y


class HOE_2_1_Module(nn.Module):
    """
    2-1 Order Equivariant Module.

    Args:
        in_features (int):  F, number of input feature channels.
        out_features (int): D, number of output feature channels.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1  = nn.Linear(5 * in_features, in_features)
        self.fc2  = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(in_features)
        self.act  = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, N, M, M, F)
        Returns:
            torch.Tensor: output tensor of shape (batch_size, N, M, D)
        """
        B, N, M, _, F = x.shape
        x = x.permute(0, 1, 4, 2, 3) # shape = (B, N, F, M, M)

        diag = torch.diagonal(x, dim1=-2, dim2=-1)
        sum_diag = diag.sum(dim=3, keepdim=True)
        sum_rows = x.sum(dim=4)
        sum_cols = x.sum(dim=3)
        sum_all  = x.sum(dim=(3, 4), keepdim=True).unsqueeze(-1)

        op1 = diag
        op2 = sum_diag.expand(-1, -1, -1, M)
        op3 = sum_rows
        op4 = sum_cols
        op5 = sum_all.expand(-1, -1, -1, M)

        x = torch.cat([op1, op2, op3, op4, op5], dim=2)
        x = x.permute(0, 1, 3, 4, 2) # shape = (B, N, M, M, F)
        
        x = self.act(self.norm(self.fc1(x)))
        y = self.fc2(x)
        return y



class MDE_Module(nn.Module):
    """
    Multidimensional Equivariant Module.

    Args:
        in_features (int): Input feature dimension.
        out_features (int): Output feature dimension.
        dim (list): List of axes for equivariant pooling.
    """

    def __init__(self, in_features, out_features, dim=[1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]):
        super(MDE_Module, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(in_features * (1 + len(dim)), out_features)

    def forward(self, x):
        """
        :param x: [bs, dim_1, ..., dim_n, d_input]  dim_1, ..., dim_n are dims exhibit multidimensional equivariance
        :return: [bs, dim_1, ..., dim_n, d_output]
        """
        pooled = [torch.mean(x, d, keepdim=True).expand_as(x) for d in self.dim]
        state = torch.cat([x] + pooled, dim=-1)
        y = self.linear(state)
        return y


class MDE_Module_LowFLOPs(nn.Module):
    """
    Multidimensional Equivariant Module (Low FLOPs version).
    Applies linear transformation before expanding to avoid redundant computations.

    Args:
        in_features (int): Input feature dimension.
        out_features (int): Output feature dimension.
        dim (list): List of axes for equivariant pooling.
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
        :param x: [bs, dim_1, ..., dim_n, d_input]  dim_1, ..., dim_n are dims exhibit multidimensional equivariance
        :return: [bs, dim_1, ..., dim_n, d_output]
        """
        y = self.linear(x)
        for i, layer in enumerate(self.layers):
            y = y + layer(torch.mean(x, self.dim[i], keepdim=True)).expand_as(y)
        return y
