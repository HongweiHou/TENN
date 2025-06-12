import torch.nn as nn
import torch
from TE_models import TE_models
from TE_models import TE_module
from TE_models.init_func import real2complex, htp, complex2real, expand_to


class PrecodingTECFP(nn.Module):
    """
     A neural network for model-driven MU-MIMO precoding combines the following sub-networks:

     MDE_network: multidimensional equivariant network
     MDI_module: multidimensional invariant module
     HOE_1_2_module: high-order equivariant module
    """

    def __init__(self, d_input, d_hidden, n_layer, MDE_dim_list):
        super(PrecodingTECFP, self).__init__()
        self.d_output = 4
        self.MDI_dim = [3]
        self.MDE_network = TE_models.MDE_Network(d_input, d_hidden, n_layer, d_hidden, MDE_dim_list)
        self.MDI_module = TE_module.MDI_Module(d_hidden, 4, self.MDI_dim)
        self.HOE_module = TE_module.HOE_1_2_Module(d_hidden, self.d_output)

    def input_layer(self, channel, noise_power):
        channel_real = complex2real(channel)
        sp = channel_real.size()
        x = torch.cat([channel_real, expand_to(noise_power, sp)], len(sp) - 1)
        return x

    def output_layer(self, x, channel, noise_power, pt):
        reciver = x[:, :, :, :, 0:2]
        weight = x[:, :, :, :, 2:4]
        prec_tensor = precoding_cf_layer(channel, reciver, weight, pt, noise_power)
        return prec_tensor

    def forward(self, channel, noise_power, pt):
        """
        :param channel: [bs, ue_num, rx_ant_num, tx_ant_num] dtype:complex
        :param noise_power: [bs 1]
        :param pt: Transmit power.
        :return: Precoding Tensor [bs, ue_num, rx_ant_num, tx_ant_num]  dtype:complex
        """
        x = self.input_layer(channel, noise_power)
        # [bs, ue_num, rx_ant_num, tx_ant_num, d_hidden]
        x = self.MDE_network(x)
        # [bs, ue_num, rx_ant_num, d_hidden]
        x = self.MDI_module(x)
        # [bs, ue_num, rx_ant_num, rx_ant_num, d_output]
        x = self.HOE_module(x)
        prec_tensor = self.output_layer(x, channel, noise_power, pt)
        return prec_tensor


def precoding_cf_layer(channel, reciver, weight, pt, noise_power):
    """
    Closed-Form Precoding Layer. Using WoodBury(WB) matrix identity to simplify the computation.
    :param channel: [bs, ue_num, rx_ant_num, tx_ant_num]
    :param reciver: [bs, rx_ant_num, rx_ant_num, 2]
    :param weight: [bs, rx_ant_num, rx_ant_num, 2]
    :param pt: [1] Transmit power
    :param noise_power: [bs, 1]
    return: Precoding Tensor [bs, ue_num, rx_ant_num, tx_ant_num]
    """
    [bs, ue_num, rx_ant_num, tx_ant_num] = channel.size()
    # [bs, ue_num, rx_ant_num, rx_ant_num]
    weight = real2complex(weight)
    receiver = real2complex(reciver)

    # [bs, ue_num*rx_ant_num, tx_ant_num]
    A_H = torch.reshape(receiver.matmul(channel), [bs, ue_num * rx_ant_num, tx_ant_num])
    HH_AH_W = torch.reshape(torch.permute(htp(channel).matmul(htp(receiver)).matmul(weight), (0, 2, 1, 3)),
                            [bs, tx_ant_num, ue_num * rx_ant_num])

    # [bs, ue_num, rx_ant_num, rx_ant_num]
    AWA = htp(receiver).matmul(weight).matmul(receiver)
    # [bs, ue_num*rx_ant_num, ue_num*rx_ant_num]
    noise_power = ((noise_power / pt) * torch.sum(torch.diagonal(torch.sum(AWA, 1), dim1=1, dim2=2), 1,
                                                  keepdim=True)).unsqueeze(
        -1) * torch.eye(ue_num * rx_ant_num).to(channel.device)

    # [bs, tx_ant_num, ue_num*rx_ant_num]
    prec_tensor = torch.linalg.solve((noise_power + A_H.matmul(HH_AH_W)).transpose(1, 2),
                                     HH_AH_W.transpose(1, 2)).transpose(
        1, 2)
    prec_tensor = torch.reshape(prec_tensor, [bs, tx_ant_num, ue_num, rx_ant_num]).permute(0, 2, 3, 1)

    prec_power = torch.abs(torch.sum(prec_tensor * torch.conj(prec_tensor), dim=[1, 2, 3], keepdim=True))
    gamma = torch.sqrt(pt / prec_power)
    prec_tensor = gamma * prec_tensor
    return prec_tensor


def cal_sum_rate_mimo(channel, prec_tensor, noise_power):
    """
    Calculate the sum-rate based on the input channel, precoding tensor, and noise power.
    :param channel: [bs, ue_num, rx_ant_num, tx_ant_num]
    :param prec_tensor: [bs, ue_num, rx_ant_num, tx_ant_num]
    :param noise_power: [bs, 1]
    :return: sum rate [bs, ue_num]
    """
    [bs, ue_num, rx_ant_num, _] = channel.shape
    prec_tensor = prec_tensor.permute((0, 3, 1, 2))
    sum_rate = torch.zeros(bs, ue_num)
    iden_mat = torch.eye(rx_ant_num).to(channel.device).repeat(bs, 1, 1)
    noise_power = expand_to(noise_power, iden_mat.size())
    for kId in range(ue_num):
        # [bs, rx_ant_num, tx_ant_num]
        k_channel = channel[:, kId, :, :]
        # [bs, tx_ant_num, rx_ant_num]
        k_prec_vec = prec_tensor[:, :, kId, :]
        # [bs, ue_num-1, tx_ant_num, rx_ant_num]
        other_prec_vec = prec_tensor[:, :, torch.arange(prec_tensor.size(2)) != kId, :].permute((0, 2, 1, 3))
        other_power_mat = torch.sum(torch.matmul(k_channel.unsqueeze(1), other_prec_vec).matmul(
            htp(torch.matmul(k_channel.unsqueeze(1), other_prec_vec), 2, 3)), 1) + noise_power * iden_mat
        inner = iden_mat + torch.inverse(other_power_mat).bmm(k_channel).bmm(k_prec_vec).bmm(htp(k_prec_vec)).bmm(
            htp(k_channel))
        sum_rate[:, kId] = torch.log2(torch.abs(torch.linalg.det(inner)))
    return sum_rate


def mmse_precoding_miso(channel, noise_power, pt):
    """
    Calculate the MMSE precoding tensor in a MISO scenario
    :param channel: [bs, ue_num, tx_ant_num]
    :param noise_power: [bs, 1]
    :param pt: [1]
    :return: Precoding Tensor [bs, ue_num, tx_ant_num]
    """
    [bs, k_num, _] = channel.shape
    noise_power = k_num * noise_power.unsqueeze(2) * (torch.eye(k_num).unsqueeze(0).repeat(bs, 1, 1)).to(channel.device)
    prec_tensor = torch.bmm(torch.conj(channel.transpose(1, 2)),
                            torch.inverse(torch.bmm(channel, torch.conj(channel.transpose(1, 2))) + noise_power))
    prec_power = torch.abs(torch.sum(prec_tensor * torch.conj(prec_tensor), [1, 2]))
    gamma = torch.sqrt(pt / prec_power)
    prec_tensor = gamma.unsqueeze(1).unsqueeze(1) * prec_tensor
    prec_tensor = prec_tensor.transpose(1, 2)
    return prec_tensor


def mmse_precoding_mimo(channel, noise_power, pt):
    """
    Calculate the MMSE precoding tensor in a MIMO scenario
    :param channel: [bs, ue_num, rx_ant_num, tx_ant_num]
    :param noise_power: [bs, 1]
    :param pt: [1]
    :return: Precoding Tensor [bs, ue_num, rx_ant_num, tx_ant_num]
    """
    [bs, k_num, rx_ant_num, tx_ant_num] = channel.shape
    channel = channel.flatten(1, 2)
    prec_tensor = mmse_precoding_miso(channel, noise_power, pt)
    prec_tensor = prec_tensor.reshape([bs, k_num, rx_ant_num, tx_ant_num])
    return prec_tensor
