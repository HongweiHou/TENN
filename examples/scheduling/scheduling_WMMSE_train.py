# -*-coding:utf-8-*-
import scheduling_func
import scheduling_models
from precoding import precoding_models
from TE_models import init_func
from TE_models import TE_models
import os

if __name__ == '__main__':
    init_func.setup_seed(3407)

    in_folder = '../data/'
    out_folder = '../save_models/scheduling_wmmse_net_out/'
    prec_folder = '../save_models/precoding_net_out/'
    prec_net_name = 'precoding_model_save'
    net_name = 'scheduling_wmmse_model_save'
    data_name = 'scheduling_data'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    batch_size, iter_num, lr_step_size = 2000, 100000, 50000
    learn_rate = 5e-4
    snr_list = [0, 10, 20, 30, 40]
    pick_num = 8
    is_gpu = True

    # define the precoding network
    with open(prec_folder+'MDE_dim_list.txt', 'r') as f:
        MDE_dim_list = f.read()
    MDE_dim_list = eval(MDE_dim_list)
    prec_net = precoding_models.PrecodingTECFP(d_input=3, d_hidden=8, n_layer=3, MDE_dim_list=MDE_dim_list)

    # define the scheduling network
    # define the network-Original
    MDE_dim_list = TE_models.generate_patterns(n_layer=4, n_dim=3, pattern='original')
    # save the MDE_dim_list
    MDE_dim_string = str(MDE_dim_list)
    with open(out_folder+'MDE_dim_list.txt', 'w') as f:
        f.write(MDE_dim_string)
    model = scheduling_models.SchedulingTEUSN(d_input=3, d_hidden=32, n_layer=4, MDE_dim_list=MDE_dim_list)

    # define the network-Pattern1
    # MDE_dim_list = TE_models.generate_patterns(n_layer=4, n_dim=3, pattern='pattern1')
    # # save the MDE_dim_list
    # MDE_dim_string = str(MDE_dim_list)
    # with open(out_folder+'MDE_dim_list.txt', 'w') as f:
    #     f.write(MDE_dim_string)
    # model = scheduling_models.SchedulingTEUSN(d_input=3, d_hidden=32, n_layer=4, MDE_dim_list=MDE_dim_list)

    # define the network-Pattern2
    # is_random = False
    # if is_random:
    #     MDE_dim_list = TE_models.generate_patterns(n_layer=4, n_dim=3, pattern='pattern2', dim_per_layer=2)
    # else:
    #     MDE_dim_list = [[[1], [2]], [[2], [3]], [[1], [3]], [[1], [2], [3]]]
    # # save the MDE_dim_list
    # MDE_dim_string = str(MDE_dim_list)
    # with open(out_folder+'MDE_dim_list.txt', 'w') as f:
    #     f.write(MDE_dim_string)
    # model = scheduling_models.SchedulingTEUSN(d_input=3, d_hidden=32, n_layer=4, MDE_dim_list=MDE_dim_list)

    # define the network-Pattern3
    # is_random = False
    # if is_random:
    #     MDE_dim_list = TE_models.generate_patterns(n_layer=4, n_dim=3, pattern='pattern3')
    # else:
    #     MDE_dim_list = [[[1]], [[2]], [[3]],[[2]]]
    # # save the MDE_dim_list
    # MDE_dim_string = str(MDE_dim_list)
    # with open(out_folder+'MDE_dim_list.txt', 'w') as f:
    #     f.write(MDE_dim_string)
    # model = scheduling_models.SchedulingTEUSN(d_input=3, d_hidden=32, n_layer=4, MDE_dim_list=MDE_dim_list)

    scheduling_wmmse_train_param = init_func.SchedulingWMMSETrainParam(model, net_name, prec_net, prec_folder,
                                                                       prec_net_name,
                                                                       pick_num, iter_num, lr_step_size, in_folder,
                                                                       out_folder,
                                                                       data_name,
                                                                       learn_rate, True, batch_size, snr_list, is_gpu)
    scheduling_func.scheduling_wmmse_train(scheduling_wmmse_train_param)
