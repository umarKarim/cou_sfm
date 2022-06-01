import torch 
import torch.nn as nn 
from .warper import InvWarper
# from options import Options



class Loss():
    def __init__(self, opts):
        super(Loss, self).__init__()
        self.ssim_wt = opts.ssim_wt 
        self.l1_wt = opts.l1_wt 
        self.geom_wt = opts.geom_wt 
        self.smooth_wt = opts.smooth_wt
        self.InvWarper = InvWarper(opts)
        self.ssim = SSIM(opts)

        # for debugging only
        self.auto_mask = []
        self.depth_mask = [] 
        self.valid_mask = []
        self.recon_tar_fr = []

    def __call__(self, in_data, out_depth, out_pose):
        curr_fr  = in_data['curr_frame']
        next_fr = in_data['next_frame']
        intrinsics = out_pose['intrinsics']
        intrinsics_inv = torch.inverse(intrinsics)
        curr_depth = out_depth['curr_depth']
        next_depth = out_depth['next_depth']
        pose_curr2nxt = out_pose['curr2nxt']
        pose_curr2nxt_inv = out_pose['curr2nxt_inv'] 

        ref_frames = [next_fr]
        ref_depths = [next_depth]
        ref_poses = [pose_curr2nxt]
        ref_poses_inv = [pose_curr2nxt_inv]

        photo_loss = 0.0
        geom_loss = 0.0
        smooth_loss = 0.0
        for ref_frame, ref_depth, ref_pose, ref_pose_inv in zip(ref_frames, ref_depths, ref_poses,
                                                                ref_poses_inv):
            photo_loss_orig, geom_loss_orig, smooth_loss_orig = self.pair_frame_loss(curr_fr, 
                                                                                     curr_depth, ref_frame, 
                                                                   ref_depth, ref_pose, intrinsics, 
                                                                   intrinsics_inv)
            photo_loss_rev, geom_loss_rev, smooth_loss_rev = self.pair_frame_loss(ref_frame, 
                                                                                  ref_depth, curr_fr, 
                                                                 curr_depth, ref_pose_inv, intrinsics, 
                                                                 intrinsics_inv)
            photo_loss += (photo_loss_orig + photo_loss_rev) 
            geom_loss += (geom_loss_orig + geom_loss_rev)
            smooth_loss += (smooth_loss_orig + smooth_loss_rev)
        net_loss = photo_loss + self.geom_wt * geom_loss + self.smooth_wt * smooth_loss 
        return {'photo_loss': photo_loss, 
                'geom_loss': geom_loss,
                'smooth_loss': smooth_loss}, net_loss 


    def pair_frame_loss(self, target_fr, target_depth, ref_frame, ref_depth, pose_tar2ref, intrinsics,
                        intrinsics_inv):
        target_fr_recon, target_depth_recon, valid_pts, d2 = self.InvWarper(ref_frame, 
                                                                       ref_depth, target_depth,
                                                                       pose_tar2ref,
                                                                       intrinsics, intrinsics_inv)

        tar_recon_diff = (target_fr - target_fr_recon).abs().clamp(0, 1)
        tar_ref_diff = (target_fr - ref_frame).abs() 
        tar_rec_depth_diff = (target_depth_recon - d2).abs()
        den = (target_depth_recon + d2)
        depth_diff = (tar_rec_depth_diff / den).clamp(0, 1)
        auto_mask, depth_mask = self.get_masks(tar_recon_diff, tar_ref_diff, depth_diff, valid_pts)
        photo_loss = self.compute_photo_loss(target_fr, target_fr_recon, tar_recon_diff, auto_mask, 
                                             depth_mask)
        geom_loss = self.compute_geom_loss(depth_diff, auto_mask)
        smooth_loss = self.compute_smooth_loss(target_depth, target_fr)
        # for intermediat results display only
        self.auto_mask = auto_mask 
        self.depth_mask = depth_mask 
        self.valid_mask = valid_pts 
        self.recon_tar_fr = target_fr_recon 
        ###################################
        return photo_loss, geom_loss, smooth_loss 

    def get_masks(self, tar_recon_diff, tar_ref_diff, depth_diff, valid_pts):
        auto_mask = tar_recon_diff.mean(dim=1, keepdim=True) < tar_ref_diff.mean(dim=1, keepdim=True)
        auto_mask = auto_mask.float()
        auto_mask = auto_mask * valid_pts
        depth_mask = (1 - depth_diff)
        return auto_mask, depth_mask 
    
    def compute_photo_loss(self, target_fr, target_fr_recon, tar_recon_diff, auto_mask, depth_mask):
        ssim_map = self.ssim(target_fr, target_fr_recon) 
        self.ssim_map = ssim_map 
        self.tar_recon_diff = tar_recon_diff
        depth_mask = depth_mask.expand_as(tar_recon_diff) 
        diff_map = depth_mask * (self.l1_wt * tar_recon_diff + self.ssim_wt * ssim_map) 
        self.diff_map = diff_map
        auto_mask = auto_mask.expand_as(diff_map)
        if auto_mask.sum() > 1e4:
            return (diff_map * auto_mask).sum() / auto_mask.sum()
        else:
            return torch.tensor(0).type_as(target_fr_recon)

    def compute_geom_loss(self, depth_diff, auto_mask):
        if auto_mask.sum() > 1e4:
            return (depth_diff * auto_mask).sum() / auto_mask.sum()
        else:
            return torch.tensor(0).type_as(depth_diff) 

    def compute_smooth_loss(self, depth, frame):
        mean_depth = depth.mean(2, True).mean(3, True) 
        n_depth = depth / (mean_depth + 1e-7) 
        grad_depth_x = torch.abs(n_depth[:, :, :, :-1] - n_depth[:, :, :, 1:])
        grad_depth_y = torch.abs(n_depth[:, :, :-1, :] - n_depth[:, :, 1:, :]) 
        grad_fr_x = torch.mean(torch.abs(n_depth[:, :, :, :-1] - n_depth[:, :, :, 1:]), 1, keepdim=True)
        grad_fr_y = torch.mean(torch.abs(n_depth[:, :, :-1, :] - n_depth[:, :, 1:, :]), 1, keepdim=True)
        wt_grad_depth_x = grad_depth_x * torch.exp(-grad_fr_x) 
        wt_grad_depth_y = grad_depth_y * torch.exp(-grad_fr_y)
        return torch.mean(wt_grad_depth_x) + torch.mean(wt_grad_depth_y)

        

class SSIM(nn.Module):
    def __init__(self, opts):
        super(SSIM, self).__init__()
        self.c1 = opts.ssim_c1
        self.c2 = opts.ssim_c2 

        self.mean_im1 = nn.AvgPool2d(3, 1)
        self.mean_im2 = nn.AvgPool2d(3, 1)
        self.mean_sig_im1 = nn.AvgPool2d(3, 1)
        self.mean_sig_im2 = nn.AvgPool2d(3, 1)
        self.mean_im1im2 = nn.AvgPool2d(3, 1)
        self.pad = nn.ReflectionPad2d(1)
    
    def forward(self, im1, im2):
        im1 = self.pad(im1)
        im2 = self.pad(im2)
        mean_im1 = self.mean_im1(im1)
        mean_im2 = self.mean_im2(im2)
        sig_im1 = self.mean_sig_im1(im1 * im1) - mean_im1 * mean_im1 
        sig_im2 = self.mean_sig_im2(im2 * im2) - mean_im2 * mean_im2 
        sig_im1_im2 = self.mean_im1im2(im1 * im2) - mean_im1 * mean_im2 
        num = (2 * mean_im1 * mean_im2 + self.c1) * (2 * sig_im1_im2 + self.c2)
        den = (mean_im1 ** 2 + mean_im2 ** 2 + self.c1) * (sig_im1 + sig_im2 + self.c2)
        return ((1 - num / den) /2.0).clamp(0, 1)


if __name__ == '__main__':
    Opts = Options()
    opts = Opts.opts 
    LossClass = Loss(opts)

    b = 10
    nan_losses = 0
    for exp in range(20):

        in_data = {}
        in_data['curr_frame'] = torch.rand(b, 3, 256, 832) 
        in_data['next_frame'] = torch.rand(b, 3, 256, 832)
        in_data['intrinsics'] = torch.rand(3, 3)
        in_data['intrinsics_inv'] = torch.inverse(in_data['intrinsics'])

        out_depth = {}
        out_depth['curr_depth'] = torch.rand(b, 1, 256, 832)
        out_depth['next_depth'] = torch.rand(b, 1, 256, 832)

        out_pose = {}
        out_pose['curr2nxt'] = torch.rand(b, 6) 
        out_pose['curr2nxt_inv'] = torch.rand(b, 6)

        loss_dict, net_loss = LossClass(in_data, out_depth, out_pose)
        print(loss_dict)
        print(net_loss)
        if torch.isnan(net_loss):
            nan_losses += 1
    print(nan_losses)




        

    
    
    