import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torchvision.transforms as transforms 
import torchvision.utils as vutils
import torch.utils.tensorboard as tb 
import torch.utils.data as data 
from dataset.Datasets import ReplayOnlineDataset, VirtualKittiWithKittiReplay
import time 
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import os 

from loss.Loss import Loss 
from options.online_train_options import Options
from options.pretrain_options import Options as PretrainOptions  
from regularization import MemRegularizer 
from networks import get_disp_network, get_pose_network 


class OnlineTrain():
    def __init__(self, opts):
        self.opts = opts          
        self.dataset_tag = opts.dataset_tag
        
        # training related options 
        self.pretrain_opts = PretrainOptions().opts
        self.batch_size = 1
        self.shuffle = False 
        self.lr = opts.lr
        self.beta1 = opts.beta1 
        self.beta2 = opts.beta2
        self.console_out = opts.console_out 
        self.save_disp = opts.save_disp  
        self.log_tb = opts.log_tensorboard 
        self.network = opts.network
        self.gpus = opts.gpus  
        
        # replay related options 
        self.apply_replay = opts.apply_replay
        self.replay_curr_fr_dir = opts.replay_curr_fr_dir 
        self.replay_next_fr_dir = opts.replay_next_fr_dir 
        self.train_loss_mean = opts.train_loss_mean 
        self.train_loss_var = opts.train_loss_var
        self.replay_model_lr = opts.replay_model_lr 
        self.replay_model_th = opts.replay_model_th
        self.replay_counter = 0

        # regularization related options 
        self.apply_mem_reg = opts.apply_mem_reg 
        self.mem_reg_wt = opts.mem_reg_wt 
        
        # intermediate results and models related options 
        self.tboard_dir = opts.tboard_dir 
        self.int_result_dir = opts.int_results_dir 
        self.save_model_dir = opts.save_model_dir 
        self.save_model_iter = opts.save_model_iter 
        self.frame_size = opts.frame_size
        self.disp_model_path = opts.disp_model_path 
        self.pose_model_path = opts.pose_model_path
        self.start_time = time.time()
        self.model_counter = 0 
        self.train_loss_path = opts.train_loss_path 

        self.tboard_dir = opts.tboard_dir 
        self.tboard_saveloss_step = opts.tboard_saveloss_step 
        self.tboard_savedepth_step = opts.tboard_savedepth_step 
        if self.log_tb:
            self.SummaryWriter = tb.writer.SummaryWriter(log_dir=self.tboard_dir)
        self.prev_flag = 'init'
        
        os.makedirs(self.int_result_dir, exist_ok=True)
        os.makedirs(self.save_model_dir, exist_ok=True)
        os.makedirs(self.replay_curr_fr_dir, exist_ok=True)
        os.makedirs(self.replay_next_fr_dir, exist_ok=True)
                
        # dataloader
        if self.dataset_tag == 'nyu' or self.dataset_tag == 'kitti':
            self.dataset = ReplayOnlineDataset(self.opts, self.pretrain_opts)
        elif self.dataset_tag == 'v_kitti':
            self.dataset = VirtualKittiWithKittiReplay(self.opts)
        self.DataLoader = data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, 
                                          num_workers=0)
            
        # loading the modules 
        if len(self.gpus) == 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:' + str(self.gpus[0]))
        self.DispModel = get_disp_network(self.network).to(self.device)
        self.PoseModel = get_pose_network(self.network).to(self.device)
        '''if self.network == 'sfml':
            from sfmlDispNet import DispResNet
            from sfmlPoseNet import PoseResNet
            self.DispModel = DispResNet().to(self.device)
            self.PoseModel = PoseResNet().to(self.device)
        elif self.network == 'diffnet':
            from diffDispNet import DispNet 
            from diffPoseNet import PoseNet 
            self.DispModel = DispNet().to(self.device)
            self.PoseModel = PoseNet().to(self.device)
        else:
            raise ValueError('Wrong network type')'''
        '''disp_module = importlib.import_module(self.disp_module)
        pose_module = importlib.import_module(self.pose_module)
        self.DispModel = disp_module.DispResNet().to(self.device)
        self.PoseModel = pose_module.PoseResNet().to(self.device)'''
        if self.disp_model_path is not None:
            self.DispModel.load_state_dict(torch.load(self.disp_model_path, map_location='cpu'))
        if self.pose_model_path is not None:
            self.PoseModel.load_state_dict(torch.load(self.pose_model_path, map_location='cpu'))
        
        if len(self.gpus) != 0:
            self.DispModel = nn.DataParallel(self.DispModel, self.gpus)
            self.PoseModel = nn.DataParallel(self.PoseModel, self.gpus)
        self.Loss = Loss(opts)

        # the optimizer 
        params_dict = [{'params': self.DispModel.parameters()},
                       {'params': self.PoseModel.parameters()}]
        # self.optim = torch.optim.SGD(params_dict, lr=self.lr)
        self.optim = torch.optim.Adam(params_dict, lr=self.lr, betas=[self.beta1, self.beta2])
        
        # memory regularizer 
        if self.apply_mem_reg:
            self.MemReg = MemRegularizer(opts, self.DispModel, self.PoseModel)

        print('Dataset for online training: {}'.format(self.dataset_tag))
        print('Total number of samples: {}'.format(len(self.DataLoader)))  
        print('Replay applied: {}'.format(self.apply_replay))  
        print('Regularization applied: {}'.format(self.apply_mem_reg))

        self.train()
        
    def train(self):
        epoch = 0
        for i, (in_data, flag, replay_flag) in enumerate(self.DataLoader):
            for key in in_data.keys():
                in_data[key] = in_data[key].to(self.device)            
            in_data['curr_frame'] = F.interpolate(in_data['curr_frame'], size=self.frame_size, 
                                                    mode='bilinear')
            in_data['next_frame'] = F.interpolate(in_data['next_frame'], size= self.frame_size, 
                                                    mode='bilinear')
            out_depth = self.get_depth(in_data) 
            out_pose = self.get_poses(in_data)
            
            self.optim.zero_grad()
            losses, net_loss = self.Loss(in_data, out_depth, out_pose)
            if self.apply_mem_reg:
                dist = (net_loss - self.train_loss_mean) ** 2 / self.train_loss_var 
                dist = dist.detach().abs()
                mem_reg_loss = 1e-2 * dist * self.MemReg.mem_regularize_loss(self.DispModel, self.PoseModel)
            else: 
                mem_reg_loss = torch.tensor(0.0)                
            losses['mem_reg_loss'] = mem_reg_loss
            tot_loss = net_loss + mem_reg_loss 
            tot_loss.backward()
            self.optim.step()
            
            if self.apply_mem_reg:
                self.MemReg.update_importance(self.DispModel, self.PoseModel)

            self.console_display(epoch, i, losses, out_depth, out_pose)
            int_disp = self.save_int_result(epoch, i, out_depth, in_data)
            self.save_model(epoch, i, flag) 
            # self.save_tboard(epoch, i, losses, in_data, flag, out_depth)
            self.replay_buffer(in_data, net_loss, replay_flag)
            self.update_replay_params(net_loss)
            if self.prev_flag != flag[0]:
                print('Domain changed from {} to {}'.format(self.prev_flag, flag[0]))
            self.prev_flag = flag[0]
        self.save_model(epoch, len(self.DataLoader) - 1, 'abc')
                
    def update_replay_params(self, net_loss):
        net_loss_d = net_loss.detach() 
        sq_diff = (net_loss_d - self.train_loss_mean) ** 2 
        if self.apply_replay or self.apply_mem_reg:
            self.train_loss_mean += self.replay_model_lr * (net_loss_d - self.train_loss_mean)
            self.train_loss_var += self.replay_model_lr * (sq_diff - self.train_loss_var)
                    
    def replay_buffer(self, in_data, net_loss, replay_flag):
        # to save or not to save the new images to the replay directory
        sq_diff = ((net_loss - self.train_loss_mean) ** 2).detach() 
        if (sq_diff > self.train_loss_var) and (not replay_flag) and (self.apply_replay):
            # save samples for replay
            with torch.no_grad():
                curr_im = in_data['curr_frame'][0, :].cpu().numpy()
                next_im = in_data['next_frame'][0, :].cpu().numpy()
                curr_im = (curr_im - curr_im.min()) / (curr_im.max() - curr_im.min())
                next_im = (next_im - next_im.min()) / (next_im.max() - next_im.min())
                im_name = ('%06d.png' % self.replay_counter)
                curr_im = np.transpose(curr_im, (1, 2, 0))
                next_im = np.transpose(next_im, (1, 2, 0))
                curr_im = cv2.cvtColor(curr_im, cv2.COLOR_RGB2BGR)
                next_im = cv2.cvtColor(next_im, cv2.COLOR_RGB2BGR)
                cv2.imwrite(self.replay_curr_fr_dir + im_name, 255 * curr_im)
                cv2.imwrite(self.replay_next_fr_dir + im_name, 255 * next_im)
                self.replay_counter += 1 

    def get_depth(self, in_data):
        out_depth = {}
        out_depth['curr_depth'] = 1.0 / self.DispModel(in_data['curr_frame'])
        out_depth['next_depth'] = 1.0 / self.DispModel(in_data['next_frame'])
        return out_depth

    def get_poses(self, in_data):
        out_pose = {}
        pose_net_out_for, foci, offsets = self.PoseModel(in_data['curr_frame'], in_data['next_frame'])
        out_pose['curr2nxt'] = pose_net_out_for[:, :6]
        intr_for = self.correct_intrinsics(foci, offsets)
        pose_net_out_rev, foci, offsets = self.PoseModel(in_data['next_frame'], in_data['curr_frame'])
        out_pose['curr2nxt_inv'] = pose_net_out_rev[:, :6]
        intr_rev = self.correct_intrinsics(foci, offsets)
        intr = (intr_for + intr_rev) / 2.0
        out_pose['intrinsics'] = intr.view(-1, 3, 3)
        return out_pose

    def correct_intrinsics(self, foci, offsets):
        h, w = self.frame_size[0], self.frame_size[1]
        b = foci.size(0)
        # applying corrections based on frame size (taken from depth from videos from wild)
        foci_out = foci.clone() 
        foci_out[:, 0] = foci[:, 0] * w 
        foci_out[:, 1] = foci[:, 1] * h 
        offsets_out = offsets.clone()  
        offsets_out[:, 0] = (offsets_out[:, 0] + 0.5) * w 
        offsets_out[:, 1] = (offsets_out[:, 1] + 0.5) * h 
        ones = torch.ones(b).to(self.device)
        zeros = torch.zeros(b).to(self.device)
        intr = torch.empty((b, 3, 3), requires_grad=False).to(self.device)
        intr[: ,0, 0] = foci_out[:, 0] 
        intr[:, 0, 1] = zeros.clone() 
        intr[:, 0, 2] = offsets_out[:, 0] 
        intr[:, 1, 0] = zeros.clone() 
        intr[:, 1, 1] = foci_out[:, 1]
        intr[:, 1, 2] = offsets_out[:, 1]
        intr[:, 2, 0] = zeros.clone() 
        intr[:, 2, 1] = zeros.clone() 
        intr[:, 2, 2] = ones.clone()
        return intr

    def console_display(self, epoch, i, losses, out_depth, out_pose):
        if i % self.console_out == 0:
            loss_list = ''
            for key, loss in losses.items():
                loss_list = loss_list + key + ':' + str(loss.item()) + ', ' 
            tt = time.time() - self.start_time
            tot_b = len(self.DataLoader)
            print('Epoch: {}, batch: {} out of {}, time: {}'.format(epoch, i, tot_b, tt))
            print(loss_list)
            print('Estimated mean: {}, estimated variance: {}'.format(self.train_loss_mean, 
                                                                      self.train_loss_var))
            print('-------------------------')
            
    def save_int_result(self, epoch, i, out_depth, in_data):
        if i % self.save_disp == 0 and self.int_result_dir is not None:
            with torch.no_grad():
                fix_disp = 1.0 / out_depth['curr_depth']
                auto_mask = self.Loss.auto_mask 
                depth_mask = self.Loss.depth_mask 
                valid_mask = self.Loss.valid_mask.float() 
                recon_fr = self.Loss.recon_tar_fr 
                fix_disp = self.gray2jet(fix_disp)
                auto_mask = self.from1ch23ch(auto_mask)
                depth_mask = self.from1ch23ch(depth_mask)
                valid_mask = self.from1ch23ch(valid_mask) 
                recon_fr = self.from1ch23ch(recon_fr)
                fix_im = (in_data['curr_frame'])[0]
                
                hor_im1 = torch.cat((fix_im, recon_fr), dim=-1)
                hor_im2 = torch.cat((valid_mask, auto_mask), dim=-1)
                hor_im3 = torch.cat((depth_mask, fix_disp), dim=-1)
                comb_im = torch.cat((hor_im1, hor_im2, hor_im3), dim=1)
                ep_str = ('%03d_' % epoch)
                iter_str = ('%05d' % i)
                im_name = self.int_result_dir + ep_str + iter_str + '.png'
                vutils.save_image(comb_im, im_name)
            print('Intermediate result saved')
            return fix_disp
    
    def from1ch23ch(self, im):
        assert len(im.size()) == 4
        new_im = im
        if im.size(1) == 1:
            new_im = im / im.max() 
            new_im = torch.cat((new_im, new_im, new_im), dim=1)
        new_im = new_im[0, :, :, :]
        return new_im

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

    def save_tboard(self, epoch, i, losses, in_data, flag, out_depth):
        global_step = epoch * len(self.DataLoader) + i
        # adding the loss(es)
        if (global_step % self.tboard_saveloss_step == 0) and self.log_tb:
            for key, loss in losses.items():
                self.SummaryWriter.add_scalar('Loss' + key, loss, global_step)
            # cat_id = self.dataset.data['all_categories'].index(flag[0])
            # self.SummaryWriter.add_scalar('Category', cat_id, global_step)
        
        # adding the input and corresponding depth 
        if (global_step % self.tboard_savedepth_step == 0) and self.log_tb:
            with torch.no_grad():
                curr_frame = in_data['curr_frame'][0, :]
                curr_frame = self.min_max(curr_frame)
                curr_depth = out_depth['curr_depth'][0, :] 
                curr_depth = self.gray2jet(curr_depth)
                self.SummaryWriter.add_image('Input', curr_frame, global_step=global_step)
                self.SummaryWriter.add_image('Depth', curr_depth, global_step=global_step)

    def min_max(self, tensor_name):
        return (tensor_name - tensor_name.min()) / (tensor_name.max() - tensor_name.min())

    def save_model(self, epoch, i, flag):
        curr_flag = flag[0]
        # global_step = epoch * len(self.DataLoader) + i 
        if curr_flag != self.prev_flag:
            count_str = ('%02d_' % self.model_counter)
            ep_str = ('%03d_' % epoch)
            iter_str = ('%05d' % i)
            disp_model_name = self.save_model_dir + count_str + self.prev_flag + '_Disp_' + ep_str + iter_str + '.pth'
            pose_model_name = self.save_model_dir + count_str + self.prev_flag + '_Pose_' + ep_str + iter_str + '.pth'
            torch.save(self.DispModel.module.state_dict(), disp_model_name)
            torch.save(self.PoseModel.module.state_dict(), pose_model_name)
            self.model_counter += 1
            print('Model saved at epoch: {}, iter {}'.format(epoch, i))
            print('========================================')



if __name__ == '__main__':
    Opts = Options()
    OnlineDepth = PreTrain(Opts.opts)
