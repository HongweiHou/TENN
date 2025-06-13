import scheduling_func
import scheduling_models
from example.precoding import precoding_models
from TE_models import init_func

if __name__ == '__main__':
    batch_size = 1000
    snr_list = [0, 10, 20, 30, 40]
    net_name = 'scheduling_wmmse_model_save'
    prec_net_name = 'precoding_model_save'
    prec_folder = '../save_models/precoding_net_out/'
    in_folder = '../data/'
    net_folder = '../save_models/scheduling_wmmse_net_out/'
    data_name = 'scheduling_data'
    is_gpu = True

    # Define the precoding network
    with open(prec_folder+'MDE_dim_list.txt', 'r') as f:
        MDE_dim_list_prec = f.read()
    MDE_dim_list_prec = eval(MDE_dim_list_prec)
    prec_net = precoding_models.PrecodingTECFP(d_input=3, d_hidden=8, n_layer=3, MDE_dim_list=MDE_dim_list_prec)
    pick_num = 8

    # Define the scheduling network
    with open(net_folder+'MDE_dim_list.txt', 'r') as f:
        MDE_dim_list = f.read()
    MDE_dim_list = eval(MDE_dim_list)
    model = scheduling_models.SchedulingTEUSN(d_input=3, d_hidden=32, n_layer=4, MDE_dim_list=MDE_dim_list)

    scheduling_wmmse_test_param = init_func.SchedulingWMMSETestParam(model, net_name, prec_net, prec_folder,
                                                                     prec_net_name,
                                                                     pick_num, in_folder, net_folder, data_name,
                                                                     batch_size, snr_list, is_gpu)
    scheduling_func.sheduling_wmmse_test(scheduling_wmmse_test_param)
