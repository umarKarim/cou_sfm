import os 
import torch 
import numpy as np
import random 

from online_train import OnlineTrain 
from options.online_train_options import Options 

def clear_directories(opts):
    if not os.path.exists(opts.save_model_dir):
        os.mkdir(opts.save_model_dir) 
    else:
        models = [opts.save_model_dir + x for x in os.listdir(opts.save_model_dir) if x.endswith('.pth')]
        [os.remove(x) for x in models]
    int_dmaps = [opts.int_results_dir + x for x in os.listdir(opts.int_results_dir) if x.endswith('.png')]
    int_res_path = opts.int_results_dir 
    curr_frames_path = opts.replay_curr_fr_dir 
    next_frames_path = opts.replay_next_fr_dir
    if os.path.exists(int_res_path):   
        int_dmaps = [opts.int_results_dir + x for x in os.listdir(opts.int_results_dir) if x.endswith('.png')]
        [os.remove(x) for x in int_dmaps] 
    if os.path.exists(curr_frames_path):
        curr_frames = [curr_frames_path + x for x in os.listdir(curr_frames_path) if x.endswith('.png')]
        [os.remove(x) for x in curr_frames] 
    if os.path.exists(next_frames_path):
        next_frames = [next_frames_path + x for x in os.listdir(next_frames_path) if x.endswith('.png')]
        [os.remove(x) for x in next_frames]


if __name__ == '__main__':
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    opts = Options().opts
    runs = opts.runs 
    
    network = 'sfml'
    opts.network = network 
    opts.dataset_tag = 'v_kitti'
    opts.disp_model_path = 'trained_models/{}/pretrained_models/kitti_only/Disp_019_09371.pth'.format(network)
    opts.pose_model_path = 'trained_models/{}/pretrained_models/kitti_only/Pose_019_09371.pth'.format(network)
    
    opts.save_model_dir = 'trained_models/{}/online_models_vkitti/'.format(network)
    opts.vkitti_exclude_domains = ['fog', 'rain']
    opts.apply_replay = False 
    opts.apply_mem_reg = False 
    clear_directories(opts)
    OnlineTrain(opts)
    print('Fine tuning done without fog and rain')
    print('===================================')
    print('===================================')

    opts.save_model_dir = 'trained_models/{}/online_models_vkitti_fog_rain/'.format(network)
    opts.vkitti_exclude_domains = []
    opts.apply_replay = False 
    opts.apply_mem_reg = False 
    clear_directories(opts)
    OnlineTrain(opts)
    print('Fine tuning done with fog and rain')
    print('===================================')
    print('===================================')

    for run in runs:
        opts.save_model_dir = 'trained_models/{}/online_models_vkitti_replay_reg_run'.format(network) + run + '/'
        opts.vkitti_exclude_domains = ['fog', 'rain']
        opts.apply_replay = True 
        opts.apply_mem_reg = True 
        clear_directories(opts)
        OnlineTrain(opts)
        print('Proposed run {} done without fog and rain'.format(run))
        print('===================================')
        print('===================================')
        
        opts.save_model_dir = 'trained_models/{}/online_models_vkitti_fog_rain_replay_reg_run'.format(network) + run + '/'
        opts.vkitti_exclude_domains = []
        opts.apply_replay = True 
        opts.apply_mem_reg = True 
        clear_directories(opts)
        OnlineTrain(opts)
        print('Proposed run {} done wit fog and rain'.format(run))
        print('===================================')
        print('===================================')
    
    
    