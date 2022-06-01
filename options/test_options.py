import argparse 

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset_tag', type=str, default='nyu')
        parser.add_argument('--eval_dir_kitti', type=str, default='trained_models/online_models_kitti_rep_reg_run1/')
        parser.add_argument('--eval_dir_nyu', type=str, default='trained_models/online_models_nyu_rep_reg_run1/')
        parser.add_argument('--results_dir_kitti', type=str, default='results/replay_online_test_loss/kitti_online/')
        parser.add_argument('--results_dir_nyu', type=str, default='results/replay_online_test_loss/nyu_online/')
        parser.add_argument('--runs', type=list, default=['1', '2', '3'])
        parser.add_argument('--metrics', type=list, default=['rmse', 'abs_rel', 'sq_rel', 'log_rmse', 'del_125', 'del_125_2'])
        parser.add_argument('--qual_results', type=bool, default=False)
        parser.add_argument('--frame_size', type=str, default="256 320")
        parser.add_argument('--shuffle', type=bool, default=False)
        parser.add_argument('--batch_size', type=int, default=1)
        
        parser.add_argument('--output_dir', type=str, default='./output_results/')
        parser.add_argument('--network', type=str, default='sfml')
        parser.add_argument('--gpu_id', type=int, default=[1])
        parser.add_argument('--train', type=bool, default=False)

        parser.add_argument('--kitti_max_depth', type=float, default=80.0)
        parser.add_argument('--nyu_max_depth', type=float, default=10.0)
        parser.add_argument('--vkitti_max_depth', type=float, default=800.0)
        parser.add_argument('--min_depth', type=float, default=1.0e-3)

        parser.add_argument('--kitti_test_in_dir', type=str, default='/dataset_temp/kitti_test_data/color/')
        parser.add_argument('--kitti_test_gt_dir', type=str, default='/dataset_temp/kitti_test_data/depth/')
        parser.add_argument('--kitti_test_output_dir', type=str, default='qual_dmaps/kitti_test_output/')
        parser.add_argument('--nyu_test_in_dir', type=str, default='/dataset_temp/nyu_test_data/color/')
        parser.add_argument('--nyu_test_gt_dir', type=str, default='/dataset_temp/nyu_test_data/depth/')
        parser.add_argument('--nyu_test_output_dir', type=str, default='qual_dmaps/nyu_test_output/')
        parser.add_argument('--vkitti_test_in_dir', type=str, default='/dataset_temp/vkitti_dataset/')
        parser.add_argument('--vkitti_test_gt_dir', type=str, default='/dataset_temp/vkitti_depth/')
        parser.add_argument('--vkitti_test_output_dir', type=str, default='qual_dmaps/vkitti_test_output/')
        parser.add_argument('--vkitti_test_perc', type=float, default=0.1)

        parser.add_argument('--kitti_test_cat_file_name', type=str, default='filenames/kitti_test_cat.npy')
        parser.add_argument('--nyu_test_cat_file_name', type=str, default='filenames/nyu_test_cat.npy')
        parser.add_argument('--kitti_eigen_test_split_file', type=str, default='filenames/kitti_eigen_test_split.txt')
        parser.add_argument('--nyu_text_file_name', type=str, default='filenames/nyu_text_file.txt')
        parser.add_argument('--kitti_pretrain_categories', type=list, default=['road', 'residential_1'])
        
        self.opts = parser.parse_args() 
        frame_size = self.opts.frame_size 
        self.opts.frame_size = [int(x) for x in frame_size.split(' ')]
        
    def __call__(self):
        return self.opts 


