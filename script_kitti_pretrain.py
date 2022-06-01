from dir_options.pretrain_options import Options 
from pretrain import PreTrain

if __name__ == '__main__':
    opts = Options().opts 
    opts.network = 'sfml'  # can be either sfml or diffnet
    opts.dataset_tag = 'kitti'
    opts.save_model_dir = 'trained_models/pretrained_models/{}/kitti_only/'.format(opts.network)
    opts.kitti_pretrain_cat_files = ['kitti_road.txt', 'kitti_residential_1.txt', 'kitti_residential_2.txt', 'kitti_city.txt', 'kitti_campus.txt']
    PreTrain(opts)
    print('Finished')