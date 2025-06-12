# -*-coding:utf-8-*-
import torch.nn as nn
import torch
from torch.nn import init
import numpy as np
import random


def init_weights(net, init_type='normal', cwg=5e-2, cbg=0.5, fwg=0.04, fbg=0.5, bwg=5e-2):
    """
    To initialize network parameters
    :param net: Predefined network
    :param init_type: Initialization type
    :param cwg: Parameter for initializing convolutional layer weights
    :param cbg: Parameter for initializing convolutional layer biases
    :param fwg: Parameter for initializing fully connected layer weights
    :param fbg: Parameter for initializing fully connected layer biases
    :param bwg: Parameter for initializing BatchNorm2d
    :return:
    """

    def init_func(net):  # define the initialization function
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                # Gaussian distribution initialization
                if init_type == 'normal':
                    init.normal_(m.weight.data, mean=0, std=cwg)
                # Xavier initialization, scaling factor applied to the variance
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=cwg)
                # Initialization function proposed for nonlinear activation functions like ReLU and Leaky ReLU
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=cwg)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, cbg)
            elif isinstance(m, nn.Linear):
                # Gaussian distribution initialization
                if init_type == 'normal':
                    init.normal_(m.weight.data, mean=0, std=fwg)
                # Xavier initialization, scaling factor applied to the variance
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=fwg)
                # Initialization function proposed for nonlinear activation functions like ReLU and Leaky ReLU
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=fwg)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, fbg)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, bwg)
                init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def htp(x, dim1=-1, dim2=-2):
    """
    Perform conjugate transpose operation on the input tensor x.
    """
    return torch.conj(x).transpose(dim1, dim2)


def expand_to(x, size):
    """
    change the shape from [bs, 1] to [bs, ..., 1]
    """
    sp1 = list(size)
    sp2 = list(size)
    sp1[1:] = [1 for _ in range(len(sp1) - 1)]
    x = torch.reshape(x, sp1)
    sp2[-1] = 1
    x = x.expand(sp2)
    return x


def setup_seed(seed):
    """
    Set random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def complex2real(x):
    """
    Convert a complex number tensor to a real number tensor.
    :param x: (complex tensor)
    :return: (real tensor),  (..., 2), the last dim consists of the real and imaginary parts of the input
    """
    sp = x.size()
    dim = len(sp)
    x = torch.cat([torch.real(x).unsqueeze(dim), torch.imag(x).unsqueeze(dim)], dim)
    return x


def real2complex(x):
    """
    Converts a real number tensor with the last dimension containing the real and imaginary parts
    to a complex number tensor.
    """
    x = x[..., 0] + 1j * x[..., 1]
    return x


class PrecodingTrainParam:
    """
    Parameter set for the `precoding_train` function.
    :param model: The predefined precoding neural network.
    :param net_name: Name of the neural network.
    :param iter_num: Number of iterations.
    :param lr_step_size: Learning rate step size.
    :param in_folder: Path to the input folder.
    :param out_folder: Path to the output folder.
    :param data_name: Filename of the training set.
    :param learn_rate: Learning rate.
    :param init: Indicates whether to initialize the network.
    :param batch_size: Batch size.
    :param snr_list: List of SNR values used for training.
    :param is_gpu: Indicates whether to use gpu.
    """

    def __init__(self, model, net_name, iter_num, lr_step_size, in_folder='../datafolder', out_folder='../savefolder',
                 data_name='../train_set', learn_rate=5e-4, init=True, batch_size=256,
                 snr_list=[0, 5, 10, 15, 20, 25, 30], is_gpu=True):
        self.model = model
        self.net_name = net_name
        self.iter_num = iter_num
        self.lr_step_size = lr_step_size
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.data_name = data_name
        self.learn_rate = learn_rate
        self.init = init
        self.batch_size = batch_size
        self.snr_list = snr_list
        self.is_gpu = is_gpu


class PrecodingTestParam:
    """
    Parameter set of the 'precoding_test' function.
    :param model: The predefined precoding neural network.
    :param net_name: Name of the neural network.
    :param in_folder:  Path to the input folder.
    :param net_folder: Path to the folder containing the neural network to load.
    :param data_name: Filename of the training set.
    :param batch_size: Batch size.
    :param snr_list: List of SNR values used for training.
    :param is_gpu: Indicates whether to use gpu.
    """

    def __init__(self, model, net_name, in_folder='../datafolder', net_folder='../netfolder',
                 data_name='../test_set', batch_size=1000,
                 snr_list=[0, 5, 10, 15, 20, 25, 30], is_gpu=True):
        self.model = model
        self.net_name = net_name
        self.in_folder = in_folder
        self.net_folder = net_folder
        self.data_name = data_name
        self.batch_size = batch_size
        self.snr_list = snr_list
        self.is_gpu = is_gpu


class SchedulingWMMSETrainParam:
    """
    Parameter set of the 'scheduling_wmmse_train' function
    :param model: The predefined scheduling neural network.
    :param net_name: Name of the scheduling neural network.
    :param prec_net: The predefined precoding neural network.
    :param prec_folder: Path to the folder containing the precoding network.
    :param prec_net_name: Name of the precoding neural network.
    :param pick_num: Number of users selected for scheduling.
    :param iter_num: Number of iterations.
    :param lr_step_size: Learning rate step size.
    :param in_folder: Path to the input folder.
    :param out_folder: Path to the output folder.
    :param data_name: Name of the training data.
    :param learn_rate: Learning rate.
    :param init: Indicates whether to initialize the network.
    :param batch_size: Batch size.
    :param snr_list: List of SNR values used for training.
    :param is_gpu: Indicates whether to use gpu.
    :return:
    """

    def __init__(self, model, net_name, prec_net, prec_folder, prec_net_name, pick_num, iter_num, lr_step_size,
                 in_folder, out_folder, data_name, learn_rate=5e-4, init=True, batch_size=256,
                 snr_list=[0, 5, 10, 15, 20, 25, 30], is_gpu=True):
        self.model = model
        self.net_name = net_name
        self.prec_net = prec_net
        self.prec_folder = prec_folder
        self.prec_net_name = prec_net_name
        self.pick_num = pick_num
        self.iter_num = iter_num
        self.lr_step_size = lr_step_size
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.data_name = data_name
        self.learn_rate = learn_rate
        self.init = init
        self.batch_size = batch_size
        self.snr_list = snr_list
        self.is_gpu = is_gpu


class SchedulingWMMSETestParam:
    """
    Parameter set of the 'scheduling_wmmse_test' function.
    :param model: The predefined scheduling neural net.
    :param net_name: Name of the scheduling neural network.
    :param prec_net: The predefined precoding neural network.
    :param prec_folder: Path to the folder containing the precoding network.
    :param prec_net_name: Name of the precoding neural network.
    :param pick_num: Number of users selected for scheduling.
    :param in_folder: Path to the input folder.
    :param net_folder: Path to the folder containing the network to load.
    :param data_name: Name of the training data.
    :param batch_size: Batch size.
    :param snr_list: List of SNR values used for training.
    :param is_gpu: Indicates whether to use gpu.
    """

    def __init__(self, model, net_name, prec_net, prec_folder, prec_net_name, pick_num, in_folder, net_folder,
                 data_name, batch_size=1000, snr_list=[0, 5, 10, 15, 20, 25, 30], is_gpu=True):
        self.model = model
        self.net_name = net_name
        self.prec_net = prec_net
        self.prec_folder = prec_folder
        self.prec_net_name = prec_net_name
        self.pick_num = pick_num
        self.in_folder = in_folder
        self.net_folder = net_folder
        self.data_name = data_name
        self.batch_size = batch_size
        self.snr_list = snr_list
        self.is_gpu = is_gpu


class SchedulingMMSETrainParam:
    """
    Parameter set of the 'scheduling_mmse_train' function.
    :param model: The predefined scheduling neural net.
    :param net_name: Name of the scheduling neural network.
    :param pick_num: Number of users selected for scheduling.
    :param iter_num: Number of iterations.
    :param lr_step_size: Learning rate step size.
    :param in_folder: Path to the input folder.
    :param out_folder: Path to the output folder.
    :param data_name: Name of the training data.
    :param learn_rate: Learning rate.
    :param init: Indicates whether to initialize the network.
    :param batch_size: Batch size.
    :param snr_list: List of SNR values used for training.
    :param is_gpu: Indicates whether to use gpu.
    """

    def __init__(self, model, net_name, pick_num, iter_num, lr_step_size, in_folder, out_folder, data_name,
                 learn_rate=5e-4, init=True, batch_size=256, snr_list=[0, 5, 10, 15, 20, 25, 30], is_gpu=True):
        self.model = model
        self.net_name = net_name
        self.pick_num = pick_num
        self.iter_num = iter_num
        self.lr_step_size = lr_step_size
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.data_name = data_name
        self.learn_rate = learn_rate
        self.init = init
        self.batch_size = batch_size
        self.snr_list = snr_list
        self.is_gpu = is_gpu


class SchedulingMMSETestParam:
    """
    Parameter set of the 'scheduling_mmse_test' function.
    :param model: The predefined scheduling neural net.
    :param net_name: Name of the scheduling neural network.
    :param net_folder: Path to the folder containing the network to load.
    :param pick_num: Number of users selected for scheduling.
    :param in_folder: Path to the input folder.
    :param data_name: Name of the training data.
    :param batch_size: Batch size.
    :param snr_list: List of SNR values used for training.
    :param is_gpu: Indicates whether to use gpu.
    """

    def __init__(self, model, net_name, net_folder, pick_num, in_folder, data_name, batch_size, snr_list, is_gpu=True):
        self.model = model
        self.net_name = net_name
        self.net_folder = net_folder
        self.pick_num = pick_num
        self.in_folder = in_folder
        self.data_name = data_name
        self.batch_size = batch_size
        self.snr_list = snr_list
        self.is_gpu = is_gpu
