from dir_options.test_options import Options 
from test_directory import EvalDirectory

import time 
import os 


if __name__ == '__main__':
    opts = Options().opts 
    runs = opts.runs 
    
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
                        
    for i in range(len(list_eval_dir_kitti)): 
        st_time = time.time()                       
        opts.eval_dir_kitti = list_eval_dir_kitti[i]
        opts.eval_dir_nyu = list_eval_dir_nyu[i]
        opts.results_dir_kitti = list_results_dir_kitti[i]
        opts.results_dir_nyu = list_results_dir_nyu[i]
        print('====================================================')
        print('Directory count: {}'.format(i))
        opts.dataset_tag = 'kitti' 
        EvalDirectory(opts)
        opts.dataset_tag = 'nyu'
        EvalDirectory(opts)
        print('Time taken: {}'.format(time.time() - st_time))
    print('Finished')