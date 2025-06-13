import torch.nn as nn
import torch

from TE_models import TE_module
from TE_models import TE_models
from TE_models.init_func import complex2real, expand_to


class SchedulingTEUSN(nn.Module):
    """
    A neural network for scheduling combines the following sub-networks:

    MDE_network: multidimensional equivariant network
    MDI_module: multidimensional invariant module
    """

    def __init__(self, d_input, d_hidden, n_layer, MDE_dim_list):
        super(SchedulingTEUSN, self).__init__()
        self.d_output = 4
        self.MDI_dim = [2, 3]
        self.MDE_network = TE_models.MDE_Network(d_input, d_hidden, n_layer, d_hidden, MDE_dim_list)
        self.MDI_module = TE_module.MDI_Module(d_hidden, 4, self.MDI_dim)
        self.fc = nn.Linear(d_hidden, 1)
        self.final = nn.Sigmoid()

    def input_layer(self, channel, noise_power):
        channel_real = complex2real(channel)
        sp = channel_real.size()
        x = torch.cat([channel_real, expand_to(noise_power, sp)], len(sp) - 1)
        return x

    def output_layer(self, x):
        eta = self.final(self.fc(x)).squeeze(-1)
        return eta

    def forward(self, channel, noise_power):
        """
        :param channel: [bs, ue_num, rx_ant_num, tx_ant_num, d_input]
        :param noise_power: [bs, 1]
        :return eta: [bs, ue_num, 1]
        """
        x = self.input_layer(channel, noise_power)
        # [bs, ue_num, rx_ant_num, tx_ant_num, d_hidden]
        x = self.MDE_network(x)
        # [bs, ue_num, d_hidden]
        x = self.MDI_module(x)
        eta = self.output_layer(x)
        return eta

