import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import torchvision
from dataset.Datasets import KittiDepthTestDataset, NYUDepthTestDataset, VirtualKittiDepthTestDataset
import torch.utils.data as data 
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt 

from networks import get_disp_network

    
    
class TestFaster(): 
    def __init__(self, opts, dataloader=None):
        self.opts = opts 
        self.dataset_tag = opts.dataset_tag 
        self.frame_size = opts.frame_size
        self.model_path = opts.model_path 
        self.gpu_id = opts.gpu_id 
        self.network = opts.network
        self.batch_size = opts.batch_size 
        self.min_depth = opts.min_depth
        self.kitti_test_cat_file_name = opts.kitti_test_cat_file_name 
        self.nyu_test_cat_file_name = opts.nyu_test_cat_file_name 
        self.kitti_test_cat= np.load(self.kitti_test_cat_file_name, allow_pickle=True).item()
        self.nyu_test_cat = np.load(self.nyu_test_cat_file_name, allow_pickle=True).item()
        self.qual_results = opts.qual_results 
        self.metrics = opts.metrics
        # The data loader 
        # getting the dataloader ready 
        if dataloader == None:
            if self.dataset_tag == 'kitti':
                dataset = KittiDepthTestDataset(self.opts)
            elif self.dataset_tag == 'nyu':
                dataset = NYUDepthTestDataset(self.opts)
            elif self.dataset_tag == 'v_kitti':
                dataset = VirtualKittiDepthTestDataset(self.opts)
            else:
                raise NameError('Dataset not found')
            self.DataLoader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, 
                                            num_workers=16)
        else:
            self.DataLoader = dataloader
            print(len(self.DataLoader))
            
        if self.dataset_tag == 'kitti':
            self.max_depth = opts.kitti_max_depth
            self.output_dir = opts.kitti_test_output_dir  
        elif self.dataset_tag == 'nyu':
            self.max_depth = opts.nyu_max_depth 
            self.output_dir = opts.nyu_test_output_dir 
        elif self.dataset_tag == 'v_kitti':
            self.max_depth = opts.vkitti_max_depth 
            self.output_dir = opts.vkitti_test_output_dir
            
        # loading the model 
        self.DispNet = get_disp_network(self.network)
        '''if self.network == 'sfml':
            from networks.sfmlDispNet import DispResNet
            self.DispNet = DispResNet()
        elif self.network == 'diffnet':
            from networks.diffDispNet import DispNet 
            self.DispNet = DispNet()
        else:
            raise ValueError('Wrong network type')'''
        self.DispNet.load_state_dict(torch.load(self.model_path))
        if self.gpu_id is not None:
            self.device = torch.device('cuda:' + str(self.gpu_id[0]))
            self.DispNet = self.DispNet.to(self.device)
            if len(self.gpu_id) > 1:   
                self.DispNet = nn.DataParallel(self.DispNet, self.gpu_id)
        else:
            self.device = torch.device('cpu')
        self.DispNet.eval()
        
    def __call__(self):
        result = self.evaluate()
        cat = {}
        net = {}
        for metric in self.metrics:
            cat[metric] = self.get_catwise_res(result[metric])
            net[metric] = np.mean(result[metric])
        return cat, net  
    
    def evaluate(self):
        comb_res = {}
        for metric in self.metrics:
            comb_res[metric] = []
        
        with torch.no_grad():
            for i, batch_data in enumerate(self.DataLoader):
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)
                batch_data['curr_frame'] = F.interpolate(batch_data['curr_frame'], self.frame_size,
                                                         mode='bilinear')
                out_depth = 1.0 / self.DispNet(batch_data['curr_frame'])
                gt = batch_data['gt']
                
                if self.qual_results:
                    self.save_result(i, batch_data, out_depth)
                    
                if self.dataset_tag == 'kitti' or self.dataset_tag == 'v_kitti':
                    out_depth = self.resize_depth(gt, out_depth)
                    out_depth = self.crop_eigen(out_depth)
                    gt = self.crop_eigen(gt)
                elif self.dataset_tag == 'nyu':
                    out_depth = self.resize_depth(gt, out_depth)
                else:
                    raise 'Unknown dataset'
                res = self.get_rmse_list(gt, out_depth)
                for key in res.keys():
                    comb_res[key] += res[key]
        return comb_res 
    
    def resize_depth(self, gt, depth):
        _, gt_h, gt_w = gt.size() 
        disp = 1.0 / (depth + 1e-6)
        disp = F.interpolate(disp, (gt_h, gt_w), mode='bilinear')
        disp = torch.squeeze(disp, 1)
        depth_out = 1 / (disp + 1e-6)
        return depth_out 
    
    def crop_eigen(self, in_im):
        _, h, w = in_im.size()
        min_h = int(0.40810811 * h)
        max_h = int(0.99189189 * h)
        min_w = int(0.03594771 * w)
        max_w = int(0.96405229 * w)
        return in_im[:, min_h: max_h, min_w: max_w]

    def get_catwise_res(self, rmse):
        rmse_dict = {}
        if self.dataset_tag == 'v_kitti':
            return rmse_dict
        elif self.dataset_tag == 'kitti':
            cat_dict = self.kitti_test_cat
        else:
            cat_dict = self.nyu_test_cat 
        rmse = np.array(rmse)
        for key, val in cat_dict.items():
            rmse_vals = rmse[val]
            rmse_dict[key] = np.mean(rmse_vals)
        return rmse_dict 
    
    def get_rmse_list(self, gt, depth):
        b_size = gt.size(0)
        rmse_list = []
        abs_rel_list = []
        sq_rel_list = []
        del_125_list = []
        del_125_2_list = []
        log_rmse_list = []
        for b in range(b_size):
            curr_depth = depth[b, :, :]
            curr_gt = gt[b, :, :]
            mask = (curr_gt > self.min_depth) * (curr_gt < self.max_depth)
            nz_depth = curr_depth[mask]
            nz_gt = curr_gt[mask] 
            depth_med = torch.median(nz_depth)
            gt_med = torch.median(nz_gt)
            scale = gt_med / (1.0 * depth_med) 
            
            rmse = torch.sqrt(((nz_gt - scale * nz_depth) ** 2).mean())
            abs_rel = (torch.abs(nz_gt - scale * nz_depth) / nz_gt).mean() 
            sq_rel = ((nz_gt - scale * nz_depth) ** 2 / nz_gt).mean()
            log_rmse = torch.sqrt(((torch.log(scale * nz_depth) - torch.log(nz_gt)) ** 2).mean())
            
            ratio_1 = scale * nz_depth / nz_gt
            ratio_2 = 1.0 / ratio_1
            thresh = torch.max(ratio_1, ratio_2)
            del_125 = ((thresh < 1.25) * 1.0).mean()
            del_125_2 = ((thresh < 1.25**2) * 1.0).mean()
            
            rmse_list.append(rmse.item())
            abs_rel_list.append(abs_rel.item())
            sq_rel_list.append(sq_rel.item())
            log_rmse_list.append(log_rmse.item())
            del_125_list.append(del_125.item())
            del_125_2_list.append(del_125_2.item())
            
        res = {'rmse': rmse_list,
               'abs_rel': abs_rel_list, 
               'sq_rel': sq_rel_list,
               'log_rmse': log_rmse_list,
               'del_125': del_125_list,
               'del_125_2': del_125_2_list} 
               
        return res 
   
    def save_result(self, i, batch_data, out_depth):
        b = batch_data['curr_frame'].size(0)
        for ii in range(b):
            curr_im_name = self.output_dir + ('%05d' % i) + '_' + str(ii) + '.png' 
            input_im = batch_data['curr_frame'][ii, :, :, :]
            depth = out_depth[ii, :, :] 
            disp = 1.0 / depth
            depth_3 = self.gray2jet(disp)
            torchvision.utils.save_image(depth_3, curr_im_name)

    def gray2jet(self, dmap):
        cmap = plt.get_cmap('magma')
        if len(dmap.size()) == 4:
            dmap_0 = dmap[0, 0, :, :].cpu().numpy()
        elif len(dmap.size()) == 3:
            dmap_0 = dmap[0, :].cpu().numpy()
        elif len(dmap.size()) == 2:
            dmap_0 = dmap.cpu().numpy()
        else:
            raise 'Wrong dimensions of depth: {}'.format(dmap.size())
        dmap_norm = (dmap_0 - dmap_0.min()) / (dmap_0.max() - dmap_0.min())
        dmap_col = cmap(dmap_norm)
        dmap_col = dmap_col[:, :, 0:3]
        dmap_col = np.transpose(dmap_col, (2, 0, 1))
        return torch.tensor(dmap_col).float().to(self.device)


