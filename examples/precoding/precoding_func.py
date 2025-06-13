import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from datetime import datetime
import os
import warnings

from precoding_models import cal_sum_rate_mimo
from TE_models import init_func


def precoding_train(precoding_train_param):
    """
    Train a precoding neural network.
    :param precoding_train_param: Parameter set for the training function.
    """
    # load parameters from precoding_train_param
    model = precoding_train_param.model
    net_name = precoding_train_param.net_name
    iter_num = precoding_train_param.iter_num
    lr_step_size = precoding_train_param.lr_step_size
    in_folder = precoding_train_param.in_folder
    out_folder = precoding_train_param.out_folder
    data_name = precoding_train_param.data_name
    learn_rate = precoding_train_param.learn_rate
    init = precoding_train_param.init
    batch_size = precoding_train_param.batch_size
    snr_list = precoding_train_param.snr_list
    is_gpu = precoding_train_param.is_gpu
    if is_gpu:
        run_device = 'cuda:0'
    else:
        run_device = 'cpu'

    # transmit power
    pt = 1
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    print_cyc = 400

    # Initialize network parameters
    if init:
        init_func.init_weights(model, init_type='kaiming')

    # load channel
    hfreq_list = loadmat(in_folder + data_name + ".mat")
    hfreq_list = hfreq_list["HList"]

    train_h_num = 55000
    start = 0
    h_train = torch.from_numpy(hfreq_list[start:(start + train_h_num), :, :].copy()).to(torch.complex64)

    index_all = np.linspace(0, train_h_num - 1, train_h_num).astype(np.int_)
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=learn_rate, weight_decay=0)
    model = model.to(run_device)
    lr_gamma = 0.1
    lr_manager = torch.optim.lr_scheduler.StepLR(optimizer, lr_step_size, gamma=lr_gamma, last_epoch=-1)

    # Train
    loss_history = np.zeros((int(iter_num / print_cyc), 2))
    tic = datetime.now()
    model.train()
    iter_id_now = 0
    for iter_cyc in range(int(iter_num / print_cyc)):
        tic_print = datetime.now()
        sum_rate_model_all = 0

        for iter_id in range(print_cyc):
            # randomly get training data
            np.random.shuffle(index_all)
            data_index = index_all[0:batch_size]
            channel = h_train[data_index, :, :, :]
            snr = torch.from_numpy(np.random.choice(snr_list, (batch_size, 1), replace=True)).to(run_device)
            noise_power = (1 / (10 ** (snr / 10)))
            channel = channel.to(run_device)
            noise_power = noise_power.to(run_device)

            # network precoding method
            model_prec_mat = model(channel, noise_power, pt)
            loss_model = - torch.mean(torch.sum(cal_sum_rate_mimo(channel, model_prec_mat, noise_power), 1))

            # back propagation
            optimizer.zero_grad()
            loss_model.backward()
            optimizer.step()

            sum_rate_model_all = sum_rate_model_all - float(loss_model.detach().cpu().numpy())

            # update learnig-rate
            lr_manager.step()

        # print
        toc_print = datetime.now()
        iter_id_now = iter_id_now + print_cyc
        print('iter:[{0}]\t' 'sumRateModel:{loss:.3f}\t' 'lr:{learnrate:f}\t' 'time:{time:.3f}secs\t'.format(
            iter_id_now, loss=sum_rate_model_all / print_cyc,
            learnrate=lr_manager.get_last_lr()[0], time=(toc_print - tic_print).total_seconds()))
        loss_history[iter_cyc, :] = [iter_id_now, round(sum_rate_model_all / print_cyc, 3)]
    toc = datetime.now()
    print('Elapsed time: %f seconds' % (toc - tic).total_seconds())

    # save
    torch.save({'state_dict': model.state_dict()}, out_folder + net_name + '.pth.tar')
    np.savetxt(out_folder + 'loss_history.txt', loss_history, fmt='%0.8f')

    # evaluate model performance on the test set
    batch_size_test = 1000
    precoding_test_set = init_func.PrecodingTestParam(model, net_name, in_folder, out_folder, data_name,
                                                      batch_size_test, snr_list)
    precoding_test(precoding_test_set)


def precoding_test(precoding_test_param):
    """
    To test a trained precoding network.
    :param precoding_test_param: The param set of the test function
    """
    # load parameters from precoding_test_param
    model = precoding_test_param.model
    net_name = precoding_test_param.net_name
    in_folder = precoding_test_param.in_folder
    net_folder = precoding_test_param.net_folder
    data_name = precoding_test_param.data_name
    batch_size = precoding_test_param.batch_size
    snr_list = precoding_test_param.snr_list
    is_gpu = precoding_test_param.is_gpu
    if is_gpu:
        run_device = 'cuda:0'
    else:
        run_device = 'cpu'

    pt = 1
    if not os.path.exists(net_folder):
        print(net_folder)
        warnings.warn('There exists no netFolder.')

    # load test data
    hfreq_list = loadmat(in_folder + data_name + ".mat")
    hfreq_list = hfreq_list["HList"]

    train_h_num = 55000
    test_h_num = 5000
    start = 0
    h_test = torch.from_numpy(hfreq_list[(start + train_h_num):(start + train_h_num + test_h_num), :, :].copy()).to(
        torch.complex64).to(run_device)

    # load trained model
    model.load_state_dict((torch.load(net_folder + net_name + '.pth.tar'))['state_dict'])
    model = model.to(run_device)

    # test the model
    model.eval()
    h_batch_num = int(test_h_num / batch_size)
    h_id_list = np.linspace(0, test_h_num - 1, test_h_num).astype(np.int_)
    tic_print = datetime.now()
    sum_rate_model_list = np.zeros([h_batch_num, len(snr_list)])
    for h_batch_id in range(h_batch_num):
        h_id = h_id_list[(h_batch_id * batch_size):((h_batch_id + 1) * batch_size)]
        channel = h_test[h_id, :, :, :]
        for snrId in range(len(snr_list)):
            snr = snr_list[snrId] * torch.ones(batch_size, 1)
            noise_power = (1 / (10 ** (snr / 10))).to(channel.device)

            # network method
            model_prec_mat = model(channel, noise_power, pt)
            loss_model = - torch.mean(torch.sum(cal_sum_rate_mimo(channel, model_prec_mat, noise_power), 1))
            sum_rate_model_list[h_batch_id, snrId] = -float(loss_model.detach().cpu().numpy())
    toc_print = datetime.now()
    # print
    sum_rate_model = np.mean(sum_rate_model_list, 0)

    print('Elapsed time: %f seconds' % (toc_print - tic_print).total_seconds())
    print(sum_rate_model)
    # plot
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
    # save
    np.savetxt(net_folder + 'sumRateModel.txt', sum_rate_model, fmt='%0.8f')

