from options.test_options import Options 
from test_directory import EvalDirectory

import time
import argparse 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='sfml')
    network = parser.parse_args().network

    opts = Options().opts 
    runs = opts.runs 
    opts.network = network  # can be sfml or diffnet 

    list_eval_dir_kitti = []
    list_eval_dir_nyu = []
    list_results_dir_kitti = []
    list_results_dir_nyu = []

    # fine tunign directories and results 
    list_eval_dir_kitti.append('trained_models/{}/online_models_kitti_ft/'.format(opts.network))
    
    list_eval_dir_nyu.append('trained_models/{}/online_models_nyu_ft/'.format(opts.network))
    
    list_results_dir_kitti.append('results/{}/online_test_loss/kitti_online/'.format(opts.network))
    
    list_results_dir_nyu.append('results/{}/online_test_loss/nyu_online/'.format(opts.network))
    
    # Proposed approach 
    for run in runs:
        list_eval_dir_kitti.append('trained_models/{}/online_models_kitti_prop_run'.format(opts.network) + run + '/')
        list_eval_dir_nyu.append('trained_models/{}/online_models_nyu_prop_run'.format(opts.network) + run + '/')
        list_results_dir_kitti.append('results/{}/prop_test_loss_run'.format(opts.network) + run + '/kitti_online/')
        list_results_dir_nyu.append('results/{}/prop_test_loss_run'.format(opts.network) + run + '/nyu_online/')
    
    
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