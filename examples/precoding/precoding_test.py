import precoding_func
import precoding_models

from TE_models import init_func

if __name__ == '__main__':
    init_func.setup_seed(3407)

    in_folder = '../data/'
    net_folder = '../save_models/precoding_net_out/'
    data_name = 'precoding_data_channel'
    net_name = 'precoding_model_save'

    batch_size = 1000
    snr_list = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    is_gpu = True

    # Define precoding network
    with open(net_folder+'MDE_dim_list.txt', 'r') as f:
        MDE_dim_list = f.read()
    MDE_dim_list = eval(MDE_dim_list)
    model = precoding_models.PrecodingTECFP(d_input=3, d_hidden=8, n_layer=3, MDE_dim_list=MDE_dim_list)

    # Test the network
    precoding_test_param = init_func.PrecodingTestParam(model, net_name, in_folder, net_folder, data_name, batch_size,
                                                        snr_list, is_gpu)
    precoding_func.precoding_test(precoding_test_param)
