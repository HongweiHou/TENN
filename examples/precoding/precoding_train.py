import precoding_func
import precoding_models
import os

from TE_models import init_func
from TE_models import TE_models

if __name__ == '__main__':
    init_func.setup_seed(3407)

    in_folder = '../data/'
    out_folder = '../save_models/precoding_net_out/'
    data_name = 'precoding_data_channel'
    net_name = 'precoding_model_save'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    batch_size, iter_num, lr_step_size = 2000, 200000, 100000
    learn_rate = 5e-4
    snr_list = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    is_gpu = True

    # define precoding-network-Original
    MDE_dim_list = TE_models.generate_patterns(n_layer=3, n_dim=3, pattern='original')
    # save the MDE_dim_list
    MDE_dim_string = str(MDE_dim_list)
    with open(out_folder+'MDE_dim_list.txt', 'w') as f:
        f.write(MDE_dim_string)
    model = precoding_models.PrecodingTECFP(d_input=3, d_hidden=8, n_layer=3, MDE_dim_list=MDE_dim_list)

    # define the network-Pattern1
    # MDE_dim_list = TE_models.generate_patterns(n_layer=3, n_dim=3, pattern='pattern1')
    # # save the MDE_dim_list
    # MDE_dim_string = str(MDE_dim_list)
    # with open(out_folder+'MDE_dim_list.txt', 'w') as f:
    #     f.write(MDE_dim_string)
    # model = precoding_models.PrecodingTECFP(d_input=3, d_hidden=8, n_layer=3, MDE_dim_list=MDE_dim_list)

    # define the network-Pattern2
    # is_random = False
    # if is_random:
    #     MDE_dim_list = TE_models.generate_patterns(n_layer=3, n_dim=3, pattern='pattern2', dim_per_layer=2)
    # else:
    #     MDE_dim_list = [[[1], [2]], [[2], [3]], [[1], [3]]]
    # # save the MDE_dim_list
    # MDE_dim_string = str(MDE_dim_list)
    # with open(out_folder+'MDE_dim_list.txt', 'w') as f:
    #     f.write(MDE_dim_string)
    # model = precoding_models.PrecodingTECFP(d_input=3, d_hidden=8, n_layer=3, MDE_dim_list=MDE_dim_list)

    # define the network-Pattern3
    # is_random = False
    # if is_random:
    #     MDE_dim_list = TE_models.generate_patterns(n_layer=3, n_dim=3, pattern='pattern3')
    # else:
    #     MDE_dim_list = [[[1]], [[2]], [[3]]]
    # # save the MDE_dim_list
    # MDE_dim_string = str(MDE_dim_list)
    # with open(out_folder+'MDE_dim_list.txt', 'w') as f:
    #     f.write(MDE_dim_string)
    # model = precoding_models.PrecodingTECFP(d_input=3, d_hidden=8, n_layer=3, MDE_dim_list=MDE_dim_list)

    # train the network
    precoding_train_param = init_func.PrecodingTrainParam(model, net_name, iter_num, lr_step_size, in_folder,
                                                          out_folder, data_name, learn_rate,
                                                          True, batch_size, snr_list, is_gpu)
    precoding_func.precoding_train(precoding_train_param)
