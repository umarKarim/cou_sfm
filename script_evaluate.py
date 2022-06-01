from evaluate import EvaluateResults 
from options.test_options import Options 

import numpy as np 
import argparse

    
def eval_function(opts):
    if opts.dataset_tag == 'nyu' or opts.dataset_tag == 'kitti':
        results = EvaluateResults(opts).complete_evaluation()
        return results        
    else:
        print('Unknown dataset tag')
        return 0

def get_mean_std(res):
    res_mean = {}
    res_std = {}
    for key in res[0].keys():
        curr_res = []
        for i in range(len(res)):
            curr_res.append(res[i][key])
        res_mean[key] = np.mean(curr_res)
        res_std[key] = np.std(curr_res)
    return list(res_mean.values()), list(res_std.values())
    
def get_mean_std_ft(res):
    res_mean = {}
    res_std = {}
    for key in res.keys():
        res_mean[key] = res[key]
        res_std[key] = 0
    return list(res_mean.values()), list(res_std.values())

def display_latex_style_w_std(mean, std):
    h = len(mean)
    w = len(mean[0])
    for r in range(h):
        for c in range(w):
            mean_str = f"{mean[r][c]:0.4f}"
            std_str = f"{std[r][c]:0.4f}"
            complete_string = mean_str + ' + ' + std_str + ' & '
            print(complete_string, end="")
        print('\\\\')

def write_latex_style_w_std(mean, std, network):
    file_name = 'sfm_{}.txt'.format(network)
    h = len(mean)
    w = len(mean[0])
    with open(file_name, 'a') as f:
        for r in range(h):
            for c in range(w):
                mean_str = f"{mean[r][c]:0.4f}"
                std_str = f"{std[r][c]:0.4f}"
                complete_string = mean_str + ' + ' + std_str + ' & '
                f.write(complete_string)
            f.write('\\\\')
            f.write('\n')
        f.write('================= \n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='sfml')
    args = parser.parse_args()
    network = args.network
     
    opts = Options().opts
    opts.network = network 
    metrics = opts.metrics
    runs = opts.runs

    list_results_dir_kitti = ['results/{}/online_test_loss/kitti_online/'.format(network)]    
    list_results_dir_nyu = ['results/{}/online_test_loss/nyu_online/'.format(network)]
    
    for run in runs:
        list_results_dir_kitti.append('results/{}/prop_test_loss_run'.format(network) + run + '/kitti_online/')
        list_results_dir_nyu.append('results/{}/prop_test_loss_run'.format(network) + run + '/nyu_online/')
    
    print('Results format: ')
    print('Finetuning: KITTI TDP, KITTI NDP')
    print('Proposed: KITTI TDP, KITTI NDP')
    
    print('Finetuning: NYU TDP, NYU NDP')
    print('Proposed: NYU TDP, NYU NDP')
    
    for curr_metric in metrics:
        kitti_res = []
        nyu_res = []
        opts.eval_metric = curr_metric
        for i in range(len(list_results_dir_kitti)): 
            opts.results_dir_kitti = list_results_dir_kitti[i]
            opts.results_dir_nyu = list_results_dir_nyu[i]
            opts.dataset_tag = 'kitti'
            kitti_res.append(eval_function(opts))
            opts.dataset_tag = 'nyu'
            nyu_res.append(eval_function(opts))
        
        tot_runs = len(runs)
        kitti_ft = kitti_res[0]
        kitti_prop_runs = kitti_res[1:tot_runs+1]
        nyu_ft = nyu_res[0]
        nyu_prop_runs = nyu_res[1:tot_runs+1]
        
        mean_arr = []
        std_arr = []
        mean, std = get_mean_std_ft(kitti_ft)
        mean_arr.append(mean)
        std_arr.append(std)
        mean, std = get_mean_std(kitti_prop_runs)
        mean_arr.append(mean)
        std_arr.append(std)
    
        mean, std = get_mean_std_ft(nyu_ft)
        mean_arr.append(mean)
        std_arr.append(std)
        mean, std = get_mean_std(nyu_prop_runs)
        mean_arr.append(mean)
        std_arr.append(std)
        
        print('Results for {}'.format(curr_metric))
        display_latex_style_w_std(mean_arr, std_arr)
        write_latex_style_w_std(mean_arr, std_arr, network)
    print('Finished')