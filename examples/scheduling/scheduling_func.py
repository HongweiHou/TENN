import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings
from scipy.io import loadmat
from scipy.io import savemat
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.precoding.precoding_models import cal_sum_rate_mimo
from examples.precoding.precoding_models import mmse_precoding_mimo
from TE_models import init_func

def scheduling_wmmse_train(scheduling_wmmse_train_param):
    """
    Train the scheduling network with labels obtained through WMMSE precoding method.
    :param scheduling_wmmse_train_param: Parameter set of the scheduling_wmmse_train function
    """
    # Load parameters from scheduling_wmmse_train_param
    model = scheduling_wmmse_train_param.model
    net_name = scheduling_wmmse_train_param.net_name
    prec_net = scheduling_wmmse_train_param.prec_net
    prec_folder = scheduling_wmmse_train_param.prec_folder
    prec_net_name = scheduling_wmmse_train_param.prec_net_name
    pick_num = scheduling_wmmse_train_param.pick_num
    iter_num = scheduling_wmmse_train_param.iter_num
    lr_step_size = scheduling_wmmse_train_param.lr_step_size
    in_folder = scheduling_wmmse_train_param.in_folder
    out_folder = scheduling_wmmse_train_param.out_folder
    data_name = scheduling_wmmse_train_param.data_name
    learn_rate = scheduling_wmmse_train_param.learn_rate
    init = scheduling_wmmse_train_param.init
    batch_size = scheduling_wmmse_train_param.batch_size
    snr_list = scheduling_wmmse_train_param.snr_list
    is_gpu = scheduling_wmmse_train_param.is_gpu
    if is_gpu:
        run_device = 'cuda:0'
    else:
        run_device = 'cpu'

    # Transmit power
    pt = 1
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    print_cyc = 400

    if init:
        init_func.init_weights(model, init_type='kaiming')

    # Load the trained precoding network
    prec_net.load_state_dict((torch.load(prec_folder + prec_net_name + '.pth.tar'))['state_dict'])

    # Load training data
    hfreq_list = loadmat(in_folder + data_name + ".mat")
    hfreq_list = hfreq_list["HList"]
    eta_label = loadmat(in_folder + data_name + "_etaWMMSE.mat")
    eta_label = eta_label["etaWMMSE"]
    train_h_num = 55000
    start = 0
    h_train = torch.from_numpy(hfreq_list[start:(start + train_h_num), :, :].copy()).to(torch.complex64)
    eta_train = torch.from_numpy(eta_label[start:(start + train_h_num), :, :].copy()).float().to(run_device)

    index_all = np.linspace(0, train_h_num - 1, train_h_num).astype(np.int_)
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=learn_rate, weight_decay=0)
    loss_func = nn.BCELoss()

    model = model.to(run_device)
    prec_net = prec_net.to(run_device)
    prec_net.eval()
    lr_gamma = 0.1
    lr_manager = torch.optim.lr_scheduler.StepLR(optimizer, lr_step_size, gamma=lr_gamma, last_epoch=-1)

    # Train
    loss_history = np.zeros((int(iter_num / print_cyc), 3))
    tic = datetime.now()
    model.train()
    iter_id_now = 0
    for iter_cyc in range(int(iter_num / print_cyc)):
        tic_print = datetime.now()
        sum_rate_model_all = 0
        loss_all = 0
        for iter_id in range(print_cyc):
            # Randomly get training data
            np.random.shuffle(index_all)
            data_index = index_all[0:batch_size]
            channel = h_train[data_index, :, :, :]
            [_, k_num, rx_ant_num, tx_ant_num] = channel.size()
            snr_id = np.random.choice(range(len(snr_list)), (batch_size, 1), replace=True)
            snr = torch.from_numpy(np.array(snr_list)[snr_id])

            snr_geteta = snr / 10
            snr_geteta = snr_geteta.to(run_device)
            noise_power = (1 / (10 ** (snr / 10)))
            channel = channel.to(run_device)
            noise_power = noise_power.to(run_device)

            # Get the label
            eta_lable = eta_train[data_index, :, :].gather(1, snr_geteta.type(torch.int64).unsqueeze(
                -1).repeat(1, 1, k_num)).squeeze(1)

            # Use network method to schedule
            eta_model = model(channel, noise_power)
            eta = eta_model.clone().detach()
            _, choose_model = torch.topk(eta, pick_num, 1, largest=True, sorted=False)
            prec_hmodel = channel.gather(1, choose_model.type(torch.int64).unsqueeze(-1).unsqueeze(-1).repeat(1, 1,
                                                                                                              rx_ant_num,
                                                                                                              tx_ant_num))
            prec_mat_model = prec_net(prec_hmodel, noise_power, pt)
            sumrate_model = torch.mean(torch.sum(cal_sum_rate_mimo(prec_hmodel, prec_mat_model, noise_power), 1))
            loss_model = loss_func(eta_model, eta_lable)

            # Back propagation
            optimizer.zero_grad()
            loss_model.backward()
            optimizer.step()

            sum_rate_model_all = sum_rate_model_all + float(sumrate_model.detach().cpu().numpy())
            loss_all = loss_all + float(loss_model.detach().cpu().numpy())

            # Update learnig-rate
            lr_manager.step()
        # Print
        toc_print = datetime.now()
        iter_id_now = iter_id_now + print_cyc
        print('iter:[{0}]\t' 'sumRateModel:{modelsumrate:.3f}\t' 'loss:{loss:.3f}\t'
              'lr:{learnrate:f}\t' 'time:{time:.3f}secs\t'.format(
            iter_id_now, modelsumrate=sum_rate_model_all / print_cyc,
            loss=loss_all / print_cyc,
            learnrate=lr_manager.get_last_lr()[0], time=(toc_print - tic_print).total_seconds()))

        loss_history[iter_cyc, :] = [iter_id_now, round(sum_rate_model_all / print_cyc, 3),
                                     round(loss_all / print_cyc, 3)]
    toc = datetime.now()
    print('Elapsed time: %f seconds' % (toc - tic).total_seconds())

    # Save
    torch.save({'state_dict': model.state_dict()}, out_folder + net_name + '.pth.tar')
    np.savetxt(out_folder + 'loss_history.txt', loss_history, fmt='%0.8f')

    # Evaluate model performance on the test set
    batch_size_test = 1000
    scheduling_wmmse_test_param = init_func.SchedulingWMMSETestParam(model, net_name, prec_net, prec_folder,
                                                                     prec_net_name,
                                                                     pick_num, in_folder, out_folder, data_name,
                                                                     batch_size_test, snr_list)
    sheduling_wmmse_test(scheduling_wmmse_test_param)


def scheduling_mmse_train(scheduling_mmse_test_param):
    """
    Train the scheduling network with labels obtained through MMSE precoding method.
    :param scheduling_mmse_test_param: Parameter set of the scheduling_mmse_train function
    """
    # Load parameters from scheduling_mmse_test_param
    model = scheduling_mmse_test_param.model
    net_name = scheduling_mmse_test_param.net_name
    pick_num = scheduling_mmse_test_param.pick_num
    iter_num = scheduling_mmse_test_param.iter_num
    lr_step_size = scheduling_mmse_test_param.lr_step_size
    in_folder = scheduling_mmse_test_param.in_folder
    out_folder = scheduling_mmse_test_param.out_folder
    data_name = scheduling_mmse_test_param.data_name
    learn_rate = scheduling_mmse_test_param.learn_rate
    init = scheduling_mmse_test_param.init
    batch_size = scheduling_mmse_test_param.batch_size
    snr_list = scheduling_mmse_test_param.snr_list
    is_gpu = scheduling_mmse_test_param.is_gpu
    if is_gpu:
        run_device = 'cuda:0'
    else:
        run_device = 'cpu'

    pt = 1
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    print_cyc = 400

    # Initialize network parameters
    if init:
        init_func.init_weights(model, init_type='kaiming')

    # Load the training data
    hfreq_list = loadmat(in_folder + data_name + ".mat")
    hfreq_list = hfreq_list["HList"]
    eta_label = loadmat(in_folder + data_name + "_etaMMSE.mat")
    eta_label = eta_label["etaMMSE"]
    train_h_num = 55000
    start = 0
    h_train = torch.from_numpy(hfreq_list[start:(start + train_h_num), :, :].copy()).to(torch.complex64)
    eta_train = torch.from_numpy(eta_label[start:(start + train_h_num), :, :].copy()).float().to(run_device)

    index_all = np.linspace(0, train_h_num - 1, train_h_num).astype(np.int_)
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=learn_rate, weight_decay=0)
    loss_func = nn.BCELoss()

    model = model.to(run_device)
    lr_gamma = 0.1
    lr_manager = torch.optim.lr_scheduler.StepLR(optimizer, lr_step_size, gamma=lr_gamma, last_epoch=-1)

    # Train
    loss_history = np.zeros((int(iter_num / print_cyc), 3))
    [_, k_num, rx_ant_num, tx_ant_num] = h_train.size()
    tic = datetime.now()
    model.train()
    iter_id_now = 0
    for iter_cyc in range(int(iter_num / print_cyc)):
        tic_print = datetime.now()
        sum_rate_model_all = 0
        loss_all = 0
        for iter_id in range(print_cyc):
            # Randomly get training data
            np.random.shuffle(index_all)
            data_index = index_all[0:batch_size]
            channel = h_train[data_index, :, :, :]
            snr_id = np.random.choice(range(len(snr_list)), (batch_size, 1), replace=True)
            snr = torch.from_numpy(np.array(snr_list)[snr_id])
            noise_power = (1 / (10 ** (snr / 10)))
            channel = channel.to(run_device)
            noise_power = noise_power.to(run_device)

            # Get label
            eta_lable = eta_train[data_index, :, :].gather(1, torch.from_numpy(snr_id).to(run_device).type(
                torch.int64).unsqueeze(
                -1).repeat(1, 1, k_num)).squeeze(1)

            # Use network to schedule
            eta_model = model(channel, noise_power)
            eta = eta_model.clone().detach()
            _, choose_model = torch.topk(eta, pick_num, 1, largest=True, sorted=False)
            prec_hmodel = channel.gather(1, choose_model.type(torch.int64).unsqueeze(-1).unsqueeze(-1).repeat(1, 1,
                                                                                                              rx_ant_num,
                                                                                                              tx_ant_num))
            prec_mat_model = mmse_precoding_mimo(prec_hmodel, noise_power, pt)
            sumrate_model = torch.mean(torch.sum(cal_sum_rate_mimo(prec_hmodel, prec_mat_model, noise_power), 1))
            loss_model = loss_func(eta_model, eta_lable)

            # Back propagation
            optimizer.zero_grad()
            loss_model.backward()
            optimizer.step()

            sum_rate_model_all = sum_rate_model_all + float(sumrate_model.detach().cpu().numpy())
            loss_all = loss_all + float(loss_model.detach().cpu().numpy())

            # Update learnig-rate
            lr_manager.step()
        # Print
        toc_print = datetime.now()
        iter_id_now = iter_id_now + print_cyc
        print('iter:[{0}]\t' 'sumRateModel:{modelsumrate:.3f}\t' 'loss:{loss:.3f}\t'
              'lr:{learnrate:f}\t' 'time:{time:.3f}secs\t'.format(
            iter_id_now, modelsumrate=sum_rate_model_all / print_cyc,
            loss=loss_all / print_cyc,
            learnrate=lr_manager.get_last_lr()[0], time=(toc_print - tic_print).total_seconds()))
        loss_history[iter_cyc, :] = [iter_id_now, round(sum_rate_model_all / print_cyc, 3),
                                     round(loss_all / print_cyc, 3)]
    toc = datetime.now()
    print('Elapsed time: %f seconds' % (toc - tic).total_seconds())

    # Save
    torch.save({'state_dict': model.state_dict()}, out_folder + net_name + '.pth.tar')
    np.savetxt(out_folder + 'loss_history.txt', loss_history, fmt='%0.8f')

    # Evaluate model performance on the test set
    batch_size_test = 1000
    scheduling_mmse_test_param = init_func.SchedulingMMSETestParam(model, net_name, out_folder, pick_num, in_folder,
                                                                   data_name, batch_size_test, snr_list)
    scheduling_mmse_test(scheduling_mmse_test_param)


def scheduling_mmse_test(scheduling_mmse_test_param):
    """
    Test the scheduling network trained using labels obtained through MMSE precoding method.
    :param scheduling_mmse_test_param: Parameter set of the scheduling_mmse_test function
    """
    # Load parameters from scheduling_mmse_test_param
    model = scheduling_mmse_test_param.model
    net_name = scheduling_mmse_test_param.net_name
    net_folder = scheduling_mmse_test_param.net_folder
    pick_num = scheduling_mmse_test_param.pick_num
    in_folder = scheduling_mmse_test_param.in_folder
    data_name = scheduling_mmse_test_param.data_name
    batch_size = scheduling_mmse_test_param.batch_size
    snr_list = scheduling_mmse_test_param.snr_list
    is_gpu = scheduling_mmse_test_param.is_gpu
    if is_gpu:
        run_device = 'cuda:0'
    else:
        run_device = 'cpu'

    pt = 1
    if not os.path.exists(net_folder):
        print(net_folder)
        warnings.warn('There exists no netFolder.')

    # Load test data
    hfreq_list = loadmat(in_folder + data_name + ".mat")
    hfreq_list = hfreq_list["HList"]
    train_h_num = 55000
    test_h_num = 5000
    start = 0
    h_test = torch.from_numpy(hfreq_list[(start + train_h_num):(start + train_h_num + test_h_num), :, :].copy()).to(
        torch.complex64).to(run_device)
    [_, k_num, rx_ant_num, tx_ant_num] = h_test.size()

    # Load the trained scheduling network
    model.load_state_dict((torch.load(net_folder + net_name + '.pth.tar'))['state_dict'])
    model = model.to(run_device)
    model.eval()

    h_batch_num = int(test_h_num / batch_size)
    h_id_list = np.linspace(0, test_h_num - 1, test_h_num).astype(np.int_)
    tic_print = datetime.now()
    sum_rate_model_list = np.zeros([h_batch_num, len(snr_list)])
    sum_rate_mmse_us = np.zeros((len(snr_list), test_h_num))

    eta_list = np.zeros((len(snr_list), test_h_num, k_num))
    # Test
    for h_batch_id in range(h_batch_num):
        h_id = h_id_list[(h_batch_id * batch_size):((h_batch_id + 1) * batch_size)]
        channel = h_test[h_id, :, :, :]
        for snr_id in range(len(snr_list)):
            snr = snr_list[snr_id] * torch.ones(batch_size, 1)
            noise_power = (1 / (10 ** (snr / 10)))
            channel = channel.to(run_device)
            noise_power = noise_power.to(run_device)

            # Test network
            eta_model = model(channel, noise_power)
            eta = eta_model.clone().detach()
            eta_list[snr_id, h_id, :] = eta_model.clone().detach().cpu().numpy()
            _, choose_model = torch.topk(eta, pick_num, 1, largest=True, sorted=False)
            prec_hmodel = channel.gather(1, choose_model.type(torch.int64).unsqueeze(-1).unsqueeze(-1).repeat(1, 1,
                                                                                                              rx_ant_num,
                                                                                                              tx_ant_num))
            prec_mat_model = mmse_precoding_mimo(prec_hmodel, noise_power, pt)
            sumrate_model = torch.mean(torch.sum(cal_sum_rate_mimo(prec_hmodel, prec_mat_model, noise_power), 1))

            sum_rate_model_list[h_batch_id, snr_id] = float(sumrate_model.detach().cpu().numpy())
    toc_print = datetime.now()
    # Print
    sum_rate_model = np.mean(sum_rate_model_list, 0)

    print('Elapsed time: %f seconds' % (toc_print - tic_print).total_seconds())
    print(sum_rate_model)
    # Plot
    plt.style.use('fivethirtyeight')
    plt.plot(snr_list, sum_rate_model, label='SR')
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel('SNR', font2)
    plt.ylabel('Sum Rate', font2)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Save
    np.savetxt(net_folder + 'sumRateModel.txt', sum_rate_model, fmt='%0.8f')
    savemat(net_folder + net_name + '_etaMMSE.mat', {'etaList': eta_list})


def sheduling_wmmse_test(scheduling_wmmse_test_param):
    """
    Test the scheduling network trained using labels obtained through WMMSE precoding method.
    :param scheduling_wmmse_test_param: Parameter set of the scheduling_wmmse_test function
    """
    # Load parameters from scheduling_wmmse_test_param
    model = scheduling_wmmse_test_param.model
    net_name = scheduling_wmmse_test_param.net_name
    prec_net = scheduling_wmmse_test_param.prec_net
    prec_folder = scheduling_wmmse_test_param.prec_folder
    prec_net_name = scheduling_wmmse_test_param.prec_net_name
    pick_num = scheduling_wmmse_test_param.pick_num
    in_folder = scheduling_wmmse_test_param.in_folder
    net_folder = scheduling_wmmse_test_param.net_folder
    data_name = scheduling_wmmse_test_param.data_name
    batch_size = scheduling_wmmse_test_param.batch_size
    snr_list = scheduling_wmmse_test_param.snr_list
    is_gpu = scheduling_wmmse_test_param.is_gpu
    if is_gpu:
        run_device = 'cuda:0'
    else:
        run_device = 'cpu'

    pt = 1
    if not os.path.exists(net_folder):
        print(net_folder)
        warnings.warn('There exists no netFolder.')

    # Load test data
    hfreq_list = loadmat(in_folder + data_name + ".mat")
    hfreq_list = hfreq_list["HList"]
    train_h_num = 55000
    test_h_num = 5000
    start = 0
    h_test = torch.from_numpy(hfreq_list[(start + train_h_num):(start + train_h_num + test_h_num), :, :].copy()).to(
        torch.complex64).to(run_device)

    # Load trained precoding network
    prec_net.load_state_dict((torch.load(prec_folder + prec_net_name + '.pth.tar'))['state_dict'])
    prec_net = prec_net.to(run_device)
    prec_net.eval()
    # Load trained scheduling network
    model.load_state_dict((torch.load(net_folder + net_name + '.pth.tar'))['state_dict'])
    model = model.to(run_device)
    model.eval()
    h_batch_num = int(test_h_num / batch_size)
    h_id_list = np.linspace(0, test_h_num - 1, test_h_num).astype(np.int_)

    tic_print = datetime.now()
    sum_rate_model_list = np.zeros([h_batch_num, len(snr_list)])
    [_, k_num, rx_ant_num, tx_ant_num] = h_test.size()
    eta_list = np.zeros((len(snr_list), test_h_num, k_num))
    # Test
    for h_batch_id in range(h_batch_num):
        h_id = h_id_list[(h_batch_id * batch_size):((h_batch_id + 1) * batch_size)]
        channel = h_test[h_id, :, :, :]
        for snr_id in range(len(snr_list)):
            snr = snr_list[snr_id] * torch.ones(batch_size, 1)
            noise_power = (1 / (10 ** (snr / 10)))
            noise_power = noise_power.to(channel.device)
            # Test the scheduling network
            eta_model = model(channel, noise_power)
            eta = eta_model.clone().detach()
            eta_list[snr_id, h_id, :] = eta_model.clone().detach().cpu().numpy()
            _, choose_model = torch.topk(eta, pick_num, 1, largest=True, sorted=False)
            prec_hmodel = channel.gather(1, choose_model.type(torch.int64).unsqueeze(-1).unsqueeze(-1).repeat(1, 1,
                                                                                                              rx_ant_num,
                                                                                                              tx_ant_num))
            prec_mat_model = prec_net(prec_hmodel, noise_power, pt)
            sumrate_model = torch.mean(torch.sum(cal_sum_rate_mimo(prec_hmodel, prec_mat_model, noise_power), 1))

            sum_rate_model_list[h_batch_id, snr_id] = float(sumrate_model.detach().cpu().numpy())
    toc_print = datetime.now()
    # Print
    sum_rate_model = np.mean(sum_rate_model_list, 0)
    print('Elapsed time: %f seconds' % (toc_print - tic_print).total_seconds())
    print(sum_rate_model)
    # Plot
    plt.style.use('fivethirtyeight')
    plt.plot(snr_list, sum_rate_model, label='SR')
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel('SNR', font2)
    plt.ylabel('Sum Rate', font2)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Save
    np.savetxt(net_folder + 'sumRateModel.txt', sum_rate_model, fmt='%0.8f')

    savemat(net_folder + net_name + '_etaWMMSE.mat', {'etaList': eta_list})

