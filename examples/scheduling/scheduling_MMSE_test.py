import scheduling_func
import scheduling_models
from TE_models import init_func

if __name__ == '__main__':
    pick_num = 8
    batch_size = 1000
    snr_list = [0, 10, 20, 30, 40]
    is_gpu = True

    net_name = 'scheduling_mmse_model_save'
    net_folder = '../save_models/scheduling_mmse_net_out/'
    data_name = 'scheduling_data'
    in_folder = '../data/'

    with open(net_folder+'MDE_dim_list.txt', 'r') as f:
        MDE_dim_list = f.read()
    MDE_dim_list = eval(MDE_dim_list)
    model = scheduling_models.SchedulingTEUSN(d_input=3, d_hidden=8, n_layer=3, MDE_dim_list=MDE_dim_list)

    scheduling_mmse_test_param = init_func.SchedulingMMSETestParam(model, net_name, net_folder, pick_num, in_folder,
                                                                   data_name, batch_size,
                                                                   snr_list, is_gpu)
    scheduling_func.scheduling_mmse_test(scheduling_mmse_test_param)
