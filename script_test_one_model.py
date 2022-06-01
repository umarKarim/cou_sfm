from test import TestFaster
from dir_options.test_options import Options 



model_path = 'trained_models/diffnet/pretrained_models/kitti_nyu/Disp_019_01127.pth'
dataset_tag = 'kitti'
network = 'diffnet'

opts = Options().opts
opts.model_path = model_path 
opts.dataset_tag = dataset_tag 
opts.network = network 
res, net_res = TestFaster(opts).__call__()
print(net_res) 

