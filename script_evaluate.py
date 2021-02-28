from evaluation import EvaluateResults 
from dir_options.test_options import Options 
from dir_options.pretrain_options import Options as OptionsPretrain
from dir_dataset.build_nyu_categories import NYUCategorySplit

import numpy as np 
import pandas as pd

def correct_results(results, opts):
    curr_dist_res = results['curr_dist']
    cross_dist_res = results['cross_dist']
    cross_domain_res = np.mean((results['pretrain_domain'], results['cross_domain']))
    online_res = results['online_adaptation_res']
    res_dict = {'Curr dist': curr_dist_res,
                'Cross_dist': cross_dist_res,
                'Online': online_res,
                'Cross domain': cross_domain_res}
    return res_dict
    
def eval_function(opts):
    if opts.dataset_tag == 'nyu' or opts.dataset_tag == 'kitti':
        results = EvaluateResults(opts).complete_evaluation()
        return correct_results(results, opts)        
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

if __name__ == '__main__':
    opts = Options().opts 
    pre_opts = OptionsPretrain().opts 
    metrics = opts.metrics
    runs = opts.runs
    nyu_train_index_end = pre_opts.nyu_train_index_end
    nyu_categories = list(NYUCategorySplit(pre_opts).__call__().keys())
    nyu_pretrain_categories = nyu_categories[:nyu_categories.index(nyu_train_index_end)]
    opts.nyu_pretrain_categories = [x[:-1] for x in nyu_pretrain_categories] 
    
    list_eval_dir_kitti = ['trained_models/online_models_kitti/']
    list_eval_dir_nyu = ['trained_models/online_models_nyu/']    
    list_results_dir_kitti = ['results/online_test_loss/kitti_online/']    
    list_results_dir_nyu = ['results/online_test_loss/nyu_online/']
    
    for run in runs:
        list_eval_dir_kitti.append('trained_models/online_models_kitti_replay_reg_run' + run + '/')
        list_eval_dir_nyu.append('trained_models/online_models_nyu_replay_reg_run' + run + '/')
        list_results_dir_kitti.append('results/replay_reg_test_loss_run' + run + '/kitti_online/')
        list_results_dir_nyu.append('results/replay_reg_test_loss_run' + run + '/nyu_online/')
    
    
    assert len(list_eval_dir_kitti) == \
        len(list_eval_dir_nyu) == \
                    len(list_results_dir_kitti) == \
                        len(list_results_dir_nyu), 'Check the number of elements'
    
    for curr_metric in metrics:
        kitti_res = []
        nyu_res = []
        opts.eval_metric = curr_metric
        for i in range(len(list_eval_dir_kitti)): 
            opts.eval_dir_kitti = list_eval_dir_kitti[i]
            opts.eval_dir_nyu = list_eval_dir_nyu[i]
            opts.results_dir_kitti = list_results_dir_kitti[i]
            opts.results_dir_nyu = list_results_dir_nyu[i]
            opts.dataset_tag = 'kitti'
            kitti_res.append(eval_function(opts))
            opts.dataset_tag = 'nyu'
            nyu_res.append(eval_function(opts))
        
        kitti_ft = kitti_res[0]
        kitti_runs = kitti_res[1:]
        nyu_ft = nyu_res[0]
        nyu_runs = nyu_res[1:]
        mean_arr = []
        std_arr = []
        mean, std = get_mean_std_ft(kitti_ft)
        mean_arr.append(mean)
        std_arr.append(std)
        mean, std = get_mean_std(kitti_runs)
        mean_arr.append(mean)
        std_arr.append(std)
        mean, std = get_mean_std_ft(nyu_ft)
        mean_arr.append(mean)
        std_arr.append(std)
        mean, std = get_mean_std(nyu_runs)
        mean_arr.append(mean)
        std_arr.append(std)
        
        print('Results for {}'.format(curr_metric))
        display_latex_style_w_std(mean_arr, std_arr)
    print('Finished')