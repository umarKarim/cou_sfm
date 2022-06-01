import argparse


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser()

        # dataset related options 
        parser.add_argument('--dataset_tag', type=str, default='kitti_nyu')
        parser.add_argument('--kitti_root', type=str, default='/hdd/local/sdb/umar/kitti_video/kitti_256/')
        parser.add_argument('--nyu_root', type=str, default='/hdd/local/sdb/umar/nyu_indoor/rectified_nyu/')
        parser.add_argument('--frame_size', type=str, default='256 320')
        parser.add_argument('--batch_size', type=int, default=12)
        parser.add_argument('--shuffle', type=bool, default=True)
        parser.add_argument('--train', type=bool, default=True)
        parser.add_argument('--nyu_train_index_end', type=str, default='indoor_balcony_')
        parser.add_argument('--kitti_pretrain_cat_files', type=list, default=['kitti_road.txt', 'kitti_residential_1.txt'])
        parser.add_argument('--nyu_cat_file_name', type=str, default='filenames/nyu_train_cat.npy')
        parser.add_argument('--kitti_cat_file_dir', type=str, default='filenames/')
        
        # optimization related options 
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--beta2', type=float, default=0.999)

        # network related options 
        parser.add_argument('--network', type=str, help='sfml or diffnet', default='diffnet')
        # parser.add_argument('--disp_module', type=str, default='DispResNet')
        # parser.add_argument('--pose_module', type=str, default='PoseResNet')
        parser.add_argument('--decoder_in_channels', type=int, default=2048)
        parser.add_argument('--disp_model_path', type=str, default=None)
        parser.add_argument('--pose_model_path', type=str, default=None)
        
        # intermediate results realted options 
        parser.add_argument('--console_out', type=int, default=50)
        parser.add_argument('--save_disp', type=int, default=50)
        parser.add_argument('--log_tensorboard', type=bool, default=True)
        parser.add_argument('--int_results_dir', type=str, default='qual_dmaps/int_results/')
        parser.add_argument('--tboard_dir', type=str, default='tboard_dir/')
        parser.add_argument('--tboard_out', type=int, default=50)

        # saving the model 
        parser.add_argument('--save_model_dir', type=str, default='trained_models/diffnet/pretrained_models/kitti_nyu/')
        parser.add_argument('--save_model_iter', type=int, default=2000)

        # training related options 
        parser.add_argument('--epochs', type=int, default=20)

        # gpus 
        parser.add_argument('--gpus', type=list, default=[0])

        # loss 
        parser.add_argument('--ssim_wt', type=float, default=0.85)
        parser.add_argument('--l1_wt', type=float, default=0.15)
        parser.add_argument('--smooth_wt', type=float, default=0.001)
        parser.add_argument('--geom_wt', type=float, default=0.5)
        parser.add_argument('--ssim_c1', type=float, default=0.01 ** 2)
        parser.add_argument('--ssim_c2', type=float, default=0.03 ** 2)


        self.opts = parser.parse_args()
        # changing the frame size 
        frame_size = self.opts.frame_size 
        frame_size = frame_size.split(' ')
        frame_size = [int(x) for x in frame_size]
        self.opts.frame_size = frame_size

    def __call__(self):
        return self.opts



if __name__ == '__main__':
    Opts = Options()
    print(Opts.opts)

