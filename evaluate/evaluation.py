import numpy as np 
import os 


class EvaluateResults():
    def __init__(self, opts):
        self.dataset_tag = opts.dataset_tag
        self.eval_metric = opts.eval_metric 
        if self.dataset_tag == 'kitti':
            self.results_dir = opts.results_dir_kitti
            npy_files = [self.results_dir + x for x in os.listdir(self.results_dir)]
            curr_dist_file = [x for x in npy_files if 'kittitrain_kittitest' in x][0]
            other_dist_file = [x for x in npy_files if 'kittitrain_nyutest' in x][0]
            self.res_curr_dist = np.load(curr_dist_file, allow_pickle=True).item()[self.eval_metric]
            self.res_other_dist = np.load(other_dist_file, allow_pickle=True).item()[self.eval_metric]
        else:
            self.results_dir = opts.results_dir_nyu
            npy_files = [self.results_dir + x for x in os.listdir(self.results_dir)]
            curr_dist_file = [x for x in npy_files if 'nyutrain_nyutest' in x][0]
            other_dist_file = [x for x in npy_files if 'nyutrain_kittitest' in x][0]
            self.res_curr_dist = np.load(curr_dist_file, allow_pickle=True).item()[self.eval_metric]
            self.res_other_dist = np.load(other_dist_file, allow_pickle=True).item()[self.eval_metric]

    def tdp(self):
        self.curr_dist_res = []
        for model_ind in range(len(self.res_curr_dist)):
            model_res = self.res_curr_dist[model_ind]['net']
            self.curr_dist_res.append(model_res)
        return self.curr_dist_res 

    def ndp(self):
        self.cross_dist_res = []
        for model_ind in range(len(self.res_other_dist)):
            model_res = self.res_other_dist[model_ind]['net']
            self.cross_dist_res.append(model_res)
        return self.cross_dist_res 

    def complete_evaluation(self):
        tdp = self.tdp()[-1]
        ndp = self.ndp()[-1]
        return {'tdp': tdp, 
                'ndp': ndp} 
    
