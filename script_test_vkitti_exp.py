# evaluation for the vkitti experiments 
from options.test_options import Options 
from test import TestFaster
from dataset import Datasets 

import numpy as np
import torch.utils.data as data 
import pandas as pd 


def display_ft_res(ft_res):
    data = np.reshape(np.array(list(ft_res.values())), (1, -1))
    df = pd.DataFrame(data)
    print(df.to_latex(float_format="%0.4f"))
    
def display_avg_res(prop_res):
    assert len(prop_res) == 3
    metrics = prop_res[0].keys()
    metric_lists = {}
    for metric in metrics:
        metric_lists[metric] = []
        for i in range(len(prop_res)):
            metric_lists[metric].append(prop_res[i][metric]) 
    metric_mean = {}
    metric_std = {}
    for metric in metrics:
        metric_mean[metric] = np.mean(metric_lists[metric])
        metric_std[metric] = np.std(metric_lists[metric])
        print(metric_std[metric])
    data = np.reshape(np.array(list(metric_mean.values())), (1, -1))
    df = pd.DataFrame(data)
    print(df.to_latex(float_format="%0.4f"))
    data = np.reshape(np.array(metric_std.values()), (1, -1))
    df = pd.DataFrame(data)
    print(df.to_latex(float_format="%0.6f"))



if __name__ == '__main__':
    opts = Options().opts 
    metrics = opts.metrics 
    network = 'sfml'
    opts.network = network 
    models = ['trained_models/{}/online_models_vkitti/05_Scene20_Disp_000_15287.pth'.format(network),
              'trained_models/{}/online_models_vkitti_replay_reg_run1/05_Scene20_Disp_000_15287.pth'.format(network),
              'trained_models/{}/online_models_vkitti_replay_reg_run2/05_Scene20_Disp_000_15287.pth'.format(network),
              'trained_models/{}/online_models_vkitti_replay_reg_run3/05_Scene20_Disp_000_15287.pth'.format(network),
              'trained_models/{}/online_models_vkitti_fog_rain/05_Scene20_Disp_000_19109.pth'.format(network),
              'trained_models/{}/online_models_vkitti_fog_rain_replay_reg_run1/05_Scene20_Disp_000_19109.pth'.format(network),
              'trained_models/{}/online_models_vkitti_fog_rain_replay_reg_run2/05_Scene20_Disp_000_19109.pth'.format(network),
              'trained_models/{}/online_models_vkitti_fog_rain_replay_reg_run3/05_Scene20_Disp_000_19109.pth'.format(network)]
    
    tags = ['kitti']
    kitti_dataset = Datasets.KittiDepthTestDataset(opts)
    kitti_dataloader = data.DataLoader(kitti_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=16)
    
    iter = 0
    tot_iter = len(models) * len(tags)
    all_res = []
    for model_path in models:
        opts.model_path = model_path 
        print('===================================================================')
        print('Iteration: {} out of: {}'.format(iter, tot_iter))
        opts.dataset_tag = 'kitti' 
        _, comb_res = TestFaster(opts, kitti_dataloader).__call__()
        all_res.append(comb_res)
        iter += 1
    np.save('combined_results_vkitti.npy', all_res)
    all_res = np.load('combined_results_vkitti.npy', allow_pickle=True)
        
    ft_res = all_res[0]
    ft_fog_rain_res = all_res[4]
    prop_res = all_res[1:4]
    prop_fog_rain_res = all_res[5:]
    print(metrics)
    display_ft_res(ft_res)
    display_avg_res(prop_res)
    display_ft_res(ft_fog_rain_res)
    display_avg_res(prop_fog_rain_res)
        
    print('Finished')
    
    
    