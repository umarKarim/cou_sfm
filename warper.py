import torch 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 


class InvWarper():
    def __init__(self, opts):
        self.h, self.w = opts.frame_size[0], opts.frame_size[1]
        self.device = torch.device('cuda:' + str(opts.gpus[0]))
        self.pix_coord = self.make_grid().float().to(self.device)   # [3, H, W]

    def make_grid(self):
        x_vals, y_vals = torch.tensor(list(range(self.w))), torch.tensor(list(range(self.h)))
        grid_y, grid_x = torch.meshgrid(y_vals, x_vals)
        grid_x, grid_y = grid_x.reshape((self.h, self.w)), grid_y.reshape((self.h, self.w))
        ones = torch.ones_like(grid_x)
        return 1.0 * torch.stack((grid_x, grid_y, ones), dim=0)   # [3, H, W]

    def __call__(self, ref_frame, ref_depth, target_depth, pose_tar2ref, intrinsics, intrinsics_inv):
        self.check_dim(ref_frame)
        self.check_dim(ref_depth) 
        self.check_dim(target_depth)

        tar_cam_coord = self.pixel2camera(target_depth, intrinsics_inv) # [B, 3, H, W]
        ref_pix_coord, ref_mat_depth = self.cam2cam2pix(tar_cam_coord, intrinsics, pose_tar2ref)
        valid_points = ref_pix_coord.abs().max(dim=-1)[0] <= 1
        valid_mask = valid_points.unsqueeze(1).float()
        tar_fr_recon = F.grid_sample(ref_frame, ref_pix_coord, align_corners=False, padding_mode='zeros') # [B, 3, H, W]
        tar_depth_recon = F.grid_sample(ref_depth, ref_pix_coord, align_corners=False, padding_mode='zeros')# [B, 1, H, W]
        return tar_fr_recon, tar_depth_recon, valid_mask, ref_mat_depth

    def cam2cam2pix(self, cam_coord, intrinsics, pose):
        proj_mat_cam2cam = self.proj4mvector(pose)      # [B, 3, 4]
        proj_mat = intrinsics @ proj_mat_cam2cam 
        rot, tr = proj_mat[:, :, :3], proj_mat[:, :, 3]

        b = cam_coord.size(0) 
        cam_coord_flat = cam_coord.reshape(b, 3, -1) 
        rot_pix_coord = rot @ cam_coord_flat 
        tr = tr.unsqueeze(-1).expand_as(rot_pix_coord)
        pix_coord = rot_pix_coord + tr 
        x = pix_coord[:, 0, :] 
        y = pix_coord[:, 1, :] 
        z = pix_coord[:, 2, :].clamp(min=1e-3)
        x_norm = x / z 
        y_norm = y / z 
        x_norm = 2 * x_norm / (self.w - 1) - 1
        y_norm = 2 * y_norm / (self.h - 1) - 1
        x_invalid = x_norm.abs() > 1.0 
        y_invalid = y_norm.abs() > 1.0 
        x_norm[x_invalid] = 2 
        y_norm[y_invalid] = 2

        pix_coord_norm = torch.stack((x_norm, y_norm), dim=2).reshape(b, self.h, self.w, 2) 
        return pix_coord_norm, z.reshape(b, 1, self.h, self.w)

    def pixel2camera(self, depth, intrinsics_inv):
        b = depth.size(0)
        depth_exp = depth.expand(b, 3, self.h, self.w)
        pix_coord_flat = self.pix_coord.reshape((1, 3, self.h * self.w)).expand(b, 3, self.h * self.w)
        mat_prod = intrinsics_inv @ pix_coord_flat.type_as(intrinsics_inv) 
        mat_prod = mat_prod.reshape((b, 3, self.h, self.w))
        result = depth_exp * mat_prod 
        return result

    def camera2pixel(self, cam_coord_in, intrinsics):
        cam_coord = cam_coord_in.reshape((-1, 3, self.h * self.w))
        b = cam_coord.size(0)
        pix_coord_flat = intrinsics @ cam_coord.type_as(intrinsics) # [B, 3, HW]
        z = pix_coord_flat[:, 2, :].clamp(min=1e-3) # [B, HW]
        x = pix_coord_flat[:, 0, :] / z     # [B, HW]
        y = pix_coord_flat[:, 1, :] / z     # [B, HW]
        x_norm = 2.0 * x / (self.w - 1) - 1.0 # [B, HW]
        y_norm = 2.0 * y / (self.h - 1) - 1.0 # [B, HW]
        
        x_invalid = ((x_norm > 1) + (x_norm < -1)).detach() # [B, HW]
        y_invalid = ((y_norm > 1) + (y_norm < -1)).detach() # [B, HW]

        x_norm[x_invalid] = 2   # [B, HW]
        y_norm[y_invalid] = 2   # [B, HW]

        sampling_coord = torch.stack((x_norm, y_norm), dim=2).reshape(b, self.h, self.w, 2)
        return sampling_coord, z.reshape(b, 1, self.h, self.w), pix_coord_flat

    def camera2camera(self, cam_coord, rot, tr):
        cam_coord_flat = cam_coord.reshape((-1, 3, self.h * self.w))    # [B, 3, HW]
        tr_exp = tr.unsqueeze(-1).expand_as(cam_coord_flat)             # [B, 3, HW]
        rot_cam_coord = rot @ cam_coord_flat                            # [B, 3, HW]
        trrot_cam_coord = rot_cam_coord + tr_exp                        
        res_cam_coord = trrot_cam_coord.reshape((-1, 3, self.h, self.w)) # [B, 3, H, W]
        return res_cam_coord  

    def proj4mvector(self, pose):
        assert pose.size(-1) == 6, 'Wrong size of the output'
        translation = pose[:, :3].unsqueeze(-1)
        rotation = pose[:, 3:]
        rot_mat = self.get_rot_mat(rotation)
        return torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    
    def get_rot_mat(self, angle):
        assert angle.size(-1) == 3, 'Wrong size for the rotation'
        ''' Taken from https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
        and https://github.com/JiawangBian/SC-SfMLearner-Release/blob/master/inverse_warp.py'''

        B = angle.size(0)
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

        cosz = torch.cos(z)
        sinz = torch.sin(z)

        zeros = z.detach()*0
        ones = zeros.detach()+1
        zmat = torch.stack([cosz, -sinz, zeros,
                            sinz,  cosz, zeros,
                            zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

        cosy = torch.cos(y)
        siny = torch.sin(y)

        ymat = torch.stack([cosy, zeros,  siny,
                            zeros,  ones, zeros,
                            -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

        cosx = torch.cos(x)
        sinx = torch.sin(x)

        xmat = torch.stack([ones, zeros, zeros,
                            zeros,  cosx, -sinx,
                            zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

        rotMat = xmat @ ymat @ zmat
        return rotMat
        

    def check_dim(self, image):
        _, _, h, w = image.size() 
        assert h == self.h and w == self.w, 'Size mismatch'

    
    
if __name__ == '__main__':
    Opts =Options()
    opts = Opts.opts
    opts.frame_size = [100, 200]
    InvWarper = InvWarper(opts)
    # InvWarper.test_pixel2camera()
    InvWarper.test_translation()

