import os 

from online_train import OnlineTrain 
from dir_options.online_train_options import Options 

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
    opts = Options().opts 
    runs = opts.runs 
    
    # finetuning only
    opts.dataset_tag = 'kitti'
    opts.save_model_dir = 'trained_models/online_models_kitti/'
    opts.apply_replay = False 
    opts.apply_mem_reg = False 
    opts.apply_adaptation = False
    clear_directories(opts)
    OnlineTrain(opts)
    
    opts.dataset_tag = 'nyu'
    opts.save_model_dir = 'trained_models/online_models_nyu/'
    opts.apply_replay = False 
    opts.apply_mem_reg = False 
    opts.apply_adaptation = False
    clear_directories(opts)
    OnlineTrain(opts)
    
    for run in runs:
        opts.dataset_tag = 'kitti'
        opts.save_model_dir = 'trained_models/online_models_kitti_replay_reg_run' + run + '/'
        opts.apply_replay = True 
        opts.apply_mem_reg = True 
        clear_directories(opts)
        OnlineTrain(opts)
        
        opts.dataset_tag = 'nyu'
        opts.save_model_dir = 'trained_models/online_models_nyu_replay_reg_run' + run + '/'
        opts.apply_replay = True 
        opts.apply_mem_reg = True  
        clear_directories(opts)
        OnlineTrain(opts)
    
    print('Finished')
    
    
    
    
    
    