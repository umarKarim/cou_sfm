from test import TestFaster
import os 
import numpy as np
from dataset.Datasets import KittiDepthTestDataset
from dataset.Datasets import NYUDepthTestDataset 
import torch.utils.data as data 



class EvalDirectory():
    def __init__(self, opts):
        self.opts = opts
        self.batch_size = opts.batch_size 
        self.dataset_tag = opts.dataset_tag # dataset over which online training was performed 
        if self.dataset_tag == 'kitti':
            self.eval_dir = opts.eval_dir_kitti 
            self.results_dir = opts.results_dir_kitti
        elif self.dataset_tag == 'nyu':
            self.eval_dir = opts.eval_dir_nyu 
            self.results_dir = opts.results_dir_nyu
        
        os.makedirs(self.results_dir, exist_ok=True)
        self.metrics = opts.metrics 
        # dataloader so that not alot of reintialization is done 
        self.opts.root = self.opts.kitti_test_in_dir 
        kitti_dataset = KittiDepthTestDataset(self.opts)
        self.opts.root = self.opts.nyu_test_in_dir 
        nyu_dataset = NYUDepthTestDataset(self.opts)
        self.KittiDataLoader = data.DataLoader(kitti_dataset, batch_size=self.batch_size, shuffle=False, 
                                            num_workers=16, pin_memory=True)
        self.NYUDataLoader = data.DataLoader(nyu_dataset, batch_size=self.batch_size, shuffle=False, 
                                             num_workers=16, pin_memory=True)
        # the models to test 
        self.model_names = [sorted([self.eval_dir + x for x in os.listdir(self.eval_dir) if 'Disp' in x])[-1]]
        
        results_dict = self.eval_kitti()
        f_name = self.results_dir + self.dataset_tag + 'train_kittitest_online.npy'
        np.save(f_name, results_dict)
        print('Kitti results are saved')
        
        results_dict = self.eval_nyu()
        f_name = self.results_dir + self.dataset_tag + 'train_nyutest_online.npy'
        np.save(f_name, results_dict)
        print('NYU results are saved')

    def eval_kitti(self):
        self.opts.dataset_tag = 'kitti'
        result_dict = {}
        for key in self.metrics:
            result_dict[key] = []
        for model in self.model_names:
            self.opts.model_path = model 
            print('Testing kitti for model: {} out of : {}'.format(model, len(self.model_names)))
            cat_res, net_res = TestFaster(self.opts, self.KittiDataLoader).__call__()    
            for metric in self.metrics:
                curr_dict = {'cat': cat_res[metric],
                             'net': net_res[metric]}
                result_dict[metric].append(curr_dict)
        return result_dict  

    def eval_nyu(self):
        self.opts.dataset_tag = 'nyu'
        result_dict = {}
        for metric in self.metrics:
            result_dict[metric] = []

        for model in self.model_names:
            self.opts.model_path = model 
            print('Testing nyu for model: {} out of: {}'.format(model, len(self.model_names)))
            cat_res, net_res = TestFaster(self.opts, self.NYUDataLoader).__call__()
            for metric in self.metrics:
                curr_dict = {'cat': cat_res[metric],
                             'net': net_res[metric]}
                result_dict[metric].append(curr_dict)
        return result_dict  


