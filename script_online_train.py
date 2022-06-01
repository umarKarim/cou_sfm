import os 
import shutil 
import torch 
import numpy as np
import random 
import argparse

from online_train import OnlineTrain 
from options.online_train_options import Options 

def clear_directories(opts):
    if not os.path.exists(opts.save_model_dir):
        os.mkdir(opts.save_model_dir) 
    else:
        models = [opts.save_model_dir + x for x in os.listdir(opts.save_model_dir) if x.endswith('.pth')]
        # [os.remove(x) for x in models]
    if os.path.exists(opts.int_results_dir):
        int_dmaps = [opts.int_results_dir + x for x in os.listdir(opts.int_results_dir) if x.endswith('.png')]
        [os.remove(x) for x in int_dmaps] 
    curr_frames_path = opts.replay_curr_fr_dir 
    next_frames_path = opts.replay_next_fr_dir
    if os.path.exists(curr_frames_path):       
        curr_frames = [curr_frames_path + x for x in os.listdir(curr_frames_path) if x.endswith('.png')]
        [os.remove(x) for x in curr_frames] 
    if os.path.exists(next_frames_path):
        next_frames = [next_frames_path + x for x in os.listdir(next_frames_path) if x.endswith('.png')]
        [os.remove(x) for x in next_frames]
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='sfml')
    parser.add_argument('--disp_model_path', type=str, 
        default='trained_models/sfml/pretrained_models/kitti_nyu/Disp_019_05419.pth')
    parser.add_argument('--pose_model_path', type=str, 
        default='trained_models/sfml/pretrained_models/kitti_nyu/Pose_019_05419.pth')
    disp_model_path = parser.parse_args().disp_model_path 
    pose_model_path = parser.parse_args().pose_model_path 
    network = parser.parse_args().network 
    opts = Options().opts 
    opts = Options().opts 
    runs = opts.runs 
    
    opts.network = network  # can be sfml or diffnet
    opts.disp_model_path = disp_model_path
    opts.pose_model_path = pose_model_path 
    
    # finetuning only
    opts.dataset_tag = 'kitti'
    opts.save_model_dir = 'trained_models/{}/online_models_kitti_ft/'.format(opts.network)
    opts.apply_replay = False 
    opts.apply_mem_reg = False 
    clear_directories(opts)
    OnlineTrain(opts)
    
    opts.dataset_tag = 'nyu'
    opts.save_model_dir = 'trained_models/{}/online_models_nyu_ft/'.format(opts.network)
    opts.apply_replay = False 
    opts.apply_mem_reg = False 
    clear_directories(opts)
    OnlineTrain(opts)


    # Proposed online training     
    for run in runs:
        print('========================================')
        print('////////////////////////////////////////')
        print('Run {} for proposed with KITTI'.format(run))
        opts.dataset_tag = 'kitti'
        opts.save_model_dir = 'trained_models/{}/online_models_kitti_prop_run'.format(opts.network) + run + '/'
        opts.apply_replay = True 
        opts.apply_mem_reg = True 
        clear_directories(opts)
        OnlineTrain(opts)
        
        print('========================================')
        print('////////////////////////////////////////')
        print('Run {} for proposed with NYU'.format(run))
        opts.dataset_tag = 'nyu'
        opts.save_model_dir = 'trained_models/{}/online_models_nyu_prop_run'.format(opts.network) + run + '/'
        opts.apply_replay = True 
        opts.apply_mem_reg = True  
        clear_directories(opts)
        OnlineTrain(opts)
    
    
    
    
    
    