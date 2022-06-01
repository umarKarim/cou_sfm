import torch
import os
import torchvision.transforms as transforms 
import PIL.Image as Image 
import random
import matplotlib.pyplot as plt
import numpy as np 
import copy
import random
import torch.nn.functional as F

from .build_nyu_categories import NYUCategorySplit
'''from pretrain_options import Options as PreOptions 
from online_train_options import Options as OnlineOptions 
from test_options import Options as TestOptions'''


class KittiDataset():
    def __init__(self, opts, has_child=False):
        self.root = opts.kitti_root
        self.has_child = has_child
        if hasattr(opts, 'gt_dir'):
            self.gt_dir = opts.gt_dir 
        self.frame_size = opts.frame_size  
        self.train = opts.train 
        self.list_transforms = []
        self.list_transforms.append(transforms.ToTensor())
        self.list_transforms.append(transforms.Normalize([0.45, 0.45, 0.45],
                                                         [0.225, 0.225, 0.225]))
        self.transforms = transforms.Compose(self.list_transforms)

        if not self.has_child:
            # acessing left videos
            self.curr_frames = []
            self.next_frames = []
            if self.train:        
                self.sequences = sorted([x for x in os.listdir(self.root) if os.path.isdir(self.root + x)])
            
            self.read_frame_names_intrinsics()
            
    def read_frame_names_intrinsics(self):
        if self.train:
            for i, s in enumerate(self.sequences):
                frame_names = os.listdir(self.root + s + '/')
                if len(frame_names) < 3:
                    continue
                frame_names = [x for x in frame_names if x.endswith('.jpg') or x.endswith('.png')]
                frame_names.sort()
                sq_curr_frames = self.complete_name(frame_names[:-1], s)
                sq_next_frames = self.complete_name(frame_names[1:], s)

                assert len(sq_curr_frames) == len(sq_next_frames)

                self.curr_frames = self.curr_frames + sq_curr_frames
                self.next_frames = self.next_frames + sq_next_frames
        else:  # Testing (one directory only) 
            frame_names = os.listdir(self.root)
            frame_names = [x for x in frame_names if x.endswith('.png') or x.endswith('.jpg')]
            frame_names = sorted(frame_names)
            self.curr_frames = self.complete_name(frame_names, None)
            self.next_frames = self.complete_name(frame_names, None)
            
        self.data = {'curr_frames': self.curr_frames,
                    'next_frames': self.next_frames}
            
        assert len(self.data['curr_frames']) == len(self.data['next_frames']), print('Size mismatch')
    
    def complete_name(self, file_names, seq_name):
        if seq_name is None:
            return [self.root + x for x in file_names]
        else:
            return [self.root + seq_name + '/' + x for x in file_names]
        
    def display_sample(self):
        i = random.randint(0, len(self.data['curr_frames']))
        curr_frame = Image.open((self.data['curr_frames'])[i])
        next_frame = Image.open(self.data['next_frames'][i])

        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(curr_frame)
        plt.subplot(2, 1, 2)
        plt.imshow(next_frame)
        plt.show()
    
    def __len__(self):
        return len(self.data['curr_frames'])

    def __getitem__(self, i):
        curr_frame = self.transforms(Image.open(self.data['curr_frames'][i]))
        next_frame = self.transforms(Image.open(self.data['next_frames'][i]))

        return {'curr_frame': curr_frame, 
                'next_frame': next_frame}


class NYUDataset(KittiDataset):
    def __init__(self, opts, has_child=False):
        self.opts = opts
        super().__init__(self.opts, has_child=True)
        self.has_child = has_child 
        self.sequences = []
        self.root = opts.root 
               
        if not self.has_child:
            self.curr_frames = []
            self.next_frames = []
            if self.train:
                self.sequences = sorted([x for x in os.listdir(self.root) if os.path.isdir(self.root + x)])
            self.read_frame_names_intrinsics()

    def read_frame_names_intrinsics(self):
        if self.train:
            for s in self.sequences:
                file_names = os.listdir(self.root + s + '/')
                if len(file_names) < 6:
                    continue
                cam_names = [x for x in file_names if x.endswith('cam.txt')]
                cam_names = sorted(cam_names)
                # reading the frame names 
                if len(cam_names) > 1:
                    frame_names = [x for  x in file_names if x.endswith('0.jpg') or x.endswith('0.png')]
                    frame_names_nxt = [x for x in file_names if x.endswith('1.jpg') or x.endswith('1.png')]
                else:
                    frame_names = [x for x in file_names if x.endswith('.jpg') or x.endswith('.png')] 
                    frame_names_nxt = copy.deepcopy(frame_names)
                frame_names = sorted(frame_names)
                frame_names_nxt = sorted(frame_names_nxt)
                sq_curr_frames = self.complete_name(frame_names, s)
                sq_next_frames = self.complete_name(frame_names_nxt, s)            

                assert len(sq_curr_frames) == len(sq_next_frames), print('Check folder: {}'.format(s))                    
                self.curr_frames = self.curr_frames + sq_curr_frames
                self.next_frames = self.next_frames + sq_next_frames
        else: 
            frame_names = os.listdir(self.root)
            frame_names = [x for x in frame_names if x.endswith('.png') or x.endswith('.jpg')]
            frame_names = sorted(frame_names)
            self.curr_frames = self.complete_name(frame_names, None)
            self.next_frames = self.complete_name(frame_names, None)
        self.data = {'curr_frames': self.curr_frames,
                    'next_frames': self.next_frames}
        assert  len(self.data['curr_frames']) == \
                len(self.data['next_frames']), print('Size mismatch')


class KITTIandNYU():
    def __init__(self, opts):
        self.opts = opts 
        self.opts.root = self.opts.kitti_root
        self.opts.im_names_npy = self.opts.im_names_npy_kitti 
        self.KITTI = KittiDataset(self.opts)
        self.opts.root = self.opts.nyu_root
        self.opts.im_names_npy = self.opts.im_names_npy_nyu
        self.NYU = NYUDataset(self.opts)
        self.transforms = self.KITTI.transforms  #can be self.NYU.transforms  
        self.nyu_len = len(self.NYU.data['curr_frames'])

    def display_sample(self):
        i = random.randint(0, self.__len__())
        data = self.__getitem__(i)
        curr_frame = data['curr_frame'].cpu().numpy() 
        next_frame = data['next_frame'].cpu().numpy()
        curr_frame = np.transpose(curr_frame, (1, 2, 0))
        next_frame = np.transpose(next_frame, (1, 2 ,0))
        
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(curr_frame)
        plt.subplot(2, 1, 2)
        plt.imshow(next_frame)
        plt.show()
    
    def __len__(self):
        return len(self.KITTI.data['curr_frames'])

    def __getitem__(self, i):
        rand_val = random.random()
        if rand_val > 0.7:
            return self.KITTI.__getitem__(i)
        else:
            i_ = i % self.nyu_len 
            return self.NYU.__getitem__(i_)


class KittiCategoryDataset(KittiDataset):
    def __init__(self, opts, has_child=False):
        self.opts = opts 
        super().__init__(self.opts, has_child=True)
        self.has_child = has_child 
        self.kitti_cat_file_dir = opts.kitti_cat_file_dir
        if not self.has_child:
            self.curr_frames = []
            self.next_frames = []
            self.seq_file_name = opts.seq_file_name 
            assert self.seq_file_name.endswith('.txt'), 'Wrong seq file name'
            all_seq_names = os.listdir(self.root)
            all_seq_names = [x for x in all_seq_names if os.path.isdir(self.root + x)]
            seq_file = open(self.kitti_cat_file_dir + self.seq_file_name, 'r')
            self.sequences = []
            for s in seq_file:
                seq_name_st = s.split('\n')[0]
                curr_sequences = [x for x in all_seq_names if x.startswith(seq_name_st)]
                self.sequences += curr_sequences
            self.sequences = list(set(self.sequences))  # to get rid of repitition 
            self.sequences = sorted(self.sequences)
            self.read_frame_names_intrinsics()
        self.data['flags'] = [self.seq_file_name.split('.')[0]] * len(self.data['curr_frames'])


class NYUCategoryDataset(NYUDataset):
    def __init__(self, opts, cat_name, has_child=False, cat_dict=None):
        self.opts = opts 
        super().__init__(self.opts, has_child=True)
        self.has_child = has_child
        self.nyu_cat_file_name = opts.nyu_cat_file_name

        # no sequence files here. Just the dictionary 
        if cat_dict is None: 
            NYUCategorySplit(self.opts).__call__()
            cat_dict = np.load(self.nyu_cat_file_name, allow_pickle=True).items()
        keys = cat_dict.keys() 
        assert cat_name in keys, 'Category name not found'
 
        if not self.has_child:
            self.sequences = cat_dict[cat_name]
            self.curr_frames = []
            self.next_frames = []
            self.read_frame_names_intrinsics()
            self.data['flags'] = [cat_name] * len(self.data['curr_frames'])


class KittiPretrain(KittiDataset):
    def __init__(self, opts):
        self.opts = opts 
        KittiDataset.__init__(self, opts, has_child=True)
        self.list_transforms = []
        self.list_transforms.append(transforms.Resize(self.frame_size))
        self.list_transforms.append(transforms.ToTensor())
        self.list_transforms.append(transforms.Normalize([0.45, 0.45, 0.45],
                                                         [0.225, 0.225, 0.225]))
        self.transforms = transforms.Compose(self.list_transforms)
        pretrain_cat_files = self.opts.kitti_pretrain_cat_files
        self.data = {'curr_frames': [], 
                    'next_frames': [], 
                    'flags': []}
        for cat_file in pretrain_cat_files:
            setattr(self.opts, 'seq_file_name', cat_file)
            inst_name = cat_file.split('.')[0]
            exec('self.' + inst_name + '= KittiCategoryDataset(self.opts)')
            self.data['curr_frames'] = \
                self.data['curr_frames'] + (eval('self.' + inst_name + ".data['curr_frames']"))
            self.data['next_frames'] = \
                self.data['next_frames'] + (eval('self.' + inst_name + ".data['next_frames']"))
            self.data['flags'] = \
                self.data['flags'] + (eval('self.' + inst_name + ".data['flags']"))
            
    def __len__(self):
        return len(self.data['curr_frames'])

    def __getitem__(self, i):
        curr_frame = self.transforms(Image.open(self.data['curr_frames'][i]))
        next_frame = self.transforms(Image.open(self.data['next_frames'][i]))
        flag = self.data['flags'][i]

        return {'curr_frame': curr_frame, 
                'next_frame': next_frame}, flag  


class NYUPretrain(NYUDataset):
    def __init__(self, opts):
        self.opts = opts 
        NYUDataset.__init__(self, self.opts, has_child=True)
        self.train_index_end = opts.nyu_train_index_end 
        self.nyu_cat_name_dict = NYUCategorySplit(self.opts).__call__()
        categories = sorted(self.nyu_cat_name_dict.keys())
        self.train_categories = categories[:categories.index(self.train_index_end)]
        self.data = {'curr_frames': [], 
                    'next_frames': [],
                    'flags': []}
        for train_category in self.train_categories:
            inst_name = train_category
            exec('self.' + inst_name + '= NYUCategoryDataset(self.opts, train_category, cat_dict= \
                self.nyu_cat_name_dict)')
            self.data['curr_frames'] = \
                self.data['curr_frames'] + (eval('self.' + inst_name + ".data['curr_frames']"))
            self.data['next_frames'] = \
                self.data['next_frames'] + (eval('self.' + inst_name + ".data['next_frames']"))
            self.data['flags'] = \
                self.data['flags'] + (eval('self.' + inst_name + ".data['flags']"))

    def __len__(self):
        return len(self.data['curr_frames'])

    def __getitem__(self, i):
        curr_frame = self.transforms(Image.open(self.data['curr_frames'][i]))
        next_frame = self.transforms(Image.open(self.data['next_frames'][i]))
        flag = self.data['flags'][i]

        return {'curr_frame': curr_frame, 
                'next_frame': next_frame, 
                'flag': flag} 


class KittiNYUPretrain():
    def __init__(self, opts):
        self.opts = opts
        self.frame_size = self.opts.frame_size 
        self.opts.root = self.opts.nyu_root 
        self.NYUPretrainDataset = NYUPretrain(self.opts)
        self.opts.root = self.opts.kitti_root
        self.kittiPretrainDataset = KittiPretrain(self.opts)
        self.frame_size = opts.frame_size 
        self.kitti_len = self.kittiPretrainDataset.__len__()
        self.nyu_len = self.NYUPretrainDataset.__len__()
        if self.kitti_len > self.nyu_len:
            self.kitti_large = True 
        else:
            self.kitti_large = False
        
        self.list_transforms = []
        self.list_transforms.append(transforms.Resize(self.frame_size))
        self.list_transforms.append(transforms.ToTensor())
        self.list_transforms.append(transforms.Normalize([0.45, 0.45, 0.45],
                                                         [0.225, 0.225, 0.225]))
        self.transforms = transforms.Compose(self.list_transforms)
        self.data = {}
        for key in self.kittiPretrainDataset.data.keys():
            self.data[key] = self.kittiPretrainDataset.data[key] + self.NYUPretrainDataset.data[key]

    def __len__(self):
        if self.kitti_large:
            return self.kitti_len 
        else:
            return self.nyu_len 

    def __getitem__(self, i):
        rand_val = random.random()
        if rand_val < 0.5:
            if self.kitti_large:
                return self.grab_sample(self.kittiPretrainDataset, i)
            else:
                i_ = i % self.kitti_len 
                return self.grab_sample(self.kittiPretrainDataset, i_)
        else:
            if self.kitti_large:
                i_ = i % self.nyu_len 
                return self.grab_sample(self.NYUPretrainDataset, i_)
            else:
                return self.grab_sample(self.NYUPretrainDataset, i)
        
    def grab_sample(self, instance, i):
        curr_frame = self.transforms(Image.open(instance.data['curr_frames'][i]))
        next_frame = self.transforms(Image.open(instance.data['next_frames'][i]))
        flag = instance.data['flags'][i]

        return {'curr_frame': curr_frame,
                'next_frame': next_frame}, flag
        
    def shuffle_data(self):
        seed = 4
        for key in self.data.keys():
            random.seed(seed)
            random.shuffle(self.data[key])

    def display_sample(self):
        i = random.randint(0, self.__len__())
        data = self.__getitem__(i)
        curr_frame = data['curr_frame'].cpu().numpy() 
        next_frame = data['next_frame'].cpu().numpy()
        curr_frame = np.transpose(curr_frame, (1, 2, 0))
        next_frame = np.transpose(next_frame, (1, 2 ,0))
        flag = data['flag']
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(curr_frame)
        plt.subplot(2, 1, 2)
        plt.imshow(next_frame)
        plt.show()


class KittiOnlineTrainDataset(KittiDataset):
    def __init__(self, opts):
        self.opts = opts 
        self.opts.root = self.opts.kitti_root
        self.frame_size = opts.frame_size 
        self.list_of_categories = []
        self.list_transforms = []
        self.list_transforms.append(transforms.ToTensor())
        self.list_transforms.append(transforms.Normalize([0.45, 0.45, 0.45],
                                                         [0.225, 0.225, 0.225]))
        self.transforms = transforms.Compose(self.list_transforms)
        KittiDataset.__init__(self, opts, has_child=True)
        online_cat_files = self.opts.kitti_online_cat_files
        self.data = {'curr_frames': [], 
                    'next_frames': [],
                    'flags': []}
        for cat_file in online_cat_files:
            setattr(self.opts, 'seq_file_name', cat_file)
            inst_name = cat_file.split('.')[0]
            self.list_of_categories += [inst_name]
            exec('self.' + inst_name + '= KittiCategoryDataset(self.opts)')
            self.data['curr_frames'] = \
                self.data['curr_frames'] + (eval('self.' + inst_name + ".data['curr_frames']"))
            self.data['next_frames'] = \
                self.data['next_frames'] + (eval('self.' + inst_name + ".data['next_frames']"))
    
            self.data['flags'] = \
                self.data['flags'] + (eval('self.' + inst_name + ".data['flags']"))
        self.data['all_categories'] = self.list_of_categories 
        
    def __len__(self):
        return len(self.data['curr_frames'])

    def __getitem__(self, i):
        curr_frame = self.transforms(Image.open(self.data['curr_frames'][i]))
        next_frame = self.transforms(Image.open(self.data['next_frames'][i]))
        flag = self.data['flags'][i]

        return {'curr_frame': curr_frame, 
                'next_frame': next_frame}, flag


class NYUOnlineTrainDataset(NYUDataset):
    def __init__(self, opts):
        self.opts = opts 
        self.opts.root = self.opts.nyu_root   
        self.frame_size = opts.frame_size 
        self.list_transforms = []
        self.list_transforms.append(transforms.ToTensor())
        self.list_transforms.append(transforms.Normalize([0.45, 0.45, 0.45],
                                                         [0.225, 0.225, 0.225]))
        self.transforms = transforms.Compose(self.list_transforms) 
        NYUDataset.__init__(self, self.opts, has_child=True)
        self.train_index_end = opts.nyu_train_index_end 
        NYUsplit = NYUCategorySplit(self.opts)
        self.nyu_cat_name_dict = NYUsplit()
        categories = sorted(self.nyu_cat_name_dict.keys())
        self.train_categories = categories[categories.index(self.train_index_end):]
        self.list_of_categories = self.train_categories 
        self.data = {'curr_frames': [], 
                    'next_frames': [],
                    'flags': []}
        for train_category in self.train_categories:
            inst_name = train_category
            exec('self.' + inst_name + '= NYUCategoryDataset(self.opts, train_category, cat_dict= \
                self.nyu_cat_name_dict)')
            self.data['curr_frames'] = \
                self.data['curr_frames'] + (eval('self.' + inst_name + ".data['curr_frames']"))
            self.data['next_frames'] = \
                self.data['next_frames'] + (eval('self.' + inst_name + ".data['next_frames']"))
            self.data['flags'] = \
                self.data['flags'] + (eval('self.' + inst_name + ".data['flags']"))
        self.data['all_categories'] = self.list_of_categories 

    def __len__(self):
        return len(self.data['curr_frames'])

    def __getitem__(self, i):
        curr_frame = self.transforms(Image.open(self.data['curr_frames'][i]))
        next_frame = self.transforms(Image.open(self.data['next_frames'][i]))
        flag = self.data['flags'][i]

        return {'curr_frame': curr_frame, 
                'next_frame': next_frame}, flag
        


class OnlineDataset():
    def __init__(self, opts):
        self.opts = opts 
        self.dataset_tag = opts.dataset_tag 
        if self.dataset_tag == 'nyu':
            self.Dataset = NYUOnlineTrainDataset(self.opts)
        elif self.dataset_tag == 'kitti':
            self.Dataset = KittiOnlineTrainDataset(self.opts)
        else:
            assert False, 'Unknown dataset tag'
        self.transforms = self.Dataset.transforms 
        
    def __len__(self):
        return self.Dataset.__len__()
    
    def __getitem__(self, i):
        curr_frame = self.transforms(Image.open(self.Dataset.data['curr_frames'][i]))
        next_frame = self.transforms(Image.open(self.Dataset.data['next_frames'][i]))
        flag = self.Dataset.data['flags'][i]
        return {'curr_frame': curr_frame, 
                'next_frame': next_frame}, flag 
        
        
class ReplayDataset():
    def __init__(self, online_opts, pretrain_opts):
        self.opts = online_opts 
        self.pretrain_opts = pretrain_opts 
        self.dataset_tag = self.opts.dataset_tag 
        self.PretrainDataset = KittiNYUPretrain(self.pretrain_opts)
        self.replay_curr_fr_dir = self.opts.replay_curr_fr_dir 
        self.replay_next_fr_dir = self.opts.replay_next_fr_dir 
        self.transforms = self.PretrainDataset.transforms 
        self.dataset_len = OnlineDataset(self.opts).__len__()
        
    def __len__(self):
        return self.dataset_len 
    
    def __getitem__(self, i):
        rand_prob = random.random()
        curr_fr = [self.replay_curr_fr_dir + x for x in os.listdir(self.replay_curr_fr_dir) if x.endswith('.png')]
        next_fr = [self.replay_next_fr_dir + x for x in os.listdir(self.replay_curr_fr_dir) if x.endswith('.png')]
        replay_flag = True
        if rand_prob < 0.5 and len(curr_fr) > 20:
            curr_fr = sorted(curr_fr)
            next_fr = sorted(next_fr)
            assert len(curr_fr) == len(next_fr), 'Check the replay directory, diff num of frames' 
            i_ = random.randint(0, len(curr_fr) - 1)
            curr_frame = self.transforms(Image.open(curr_fr[i_]))
            next_frame = self.transforms(Image.open(next_fr[i_]))
        else:
            i_ = random.randint(0, len(self.PretrainDataset))
            curr_frame = self.transforms(Image.open(self.PretrainDataset.data['curr_frames'][i_]))
            next_frame = self.transforms(Image.open(self.PretrainDataset.data['next_frames'][i_]))
        return {'curr_frame': curr_frame, 
                'next_frame': next_frame}, replay_flag 


class ReplayOnlineDataset():
    def __init__(self, online_opts, pretrain_opts):
        self.opts = online_opts
        self.pretrain_opts = pretrain_opts 
        self.dataset_tag = online_opts.dataset_tag 
        self.apply_replay = online_opts.apply_replay 
        self.comoda = online_opts.comoda
        if self.dataset_tag == 'kitti':
            self.OnlineDataset = KittiOnlineTrainDataset(self.opts)
        else:
            self.OnlineDataset = NYUOnlineTrainDataset(self.opts) 
        self.PretrainDataset = KittiNYUPretrain(self.pretrain_opts)
        self.PretrainDataset.shuffle_data()
        self.transforms = self.PretrainDataset.transforms 
        self.replay_curr_fr_dir = self.opts.replay_curr_fr_dir
        self.replay_next_fr_dir = self.opts.replay_next_fr_dir
        self.pretrain_prob = self.opts.pretrain_prob 
        
        self.replay_online_rand_val = np.random.rand(self.__len__())
        self.replay_pretrain_rand_val = np.random.rand(self.__len__())
        self.max_pretrain_frames = self.PretrainDataset.__len__()
        self.max_replay_frames = 20000 
        self.rand_pretrain_inds = np.random.rand(self.__len__()) * (self.max_pretrain_frames - 1)
        self.rand_replay_inds = np.random.rand(self.__len__()) * (self.max_replay_frames - 1) 
        self.rand_pretrain_inds = self.rand_pretrain_inds.astype(int) 
        self.rand_replay_inds = self.rand_replay_inds.astype(int)
        
    def __len__(self):
        return len(self.OnlineDataset)  # 20000 40k images are close to 0.75G space 

    def __getitem__(self, i):
        flag = self.get_data_flag(i)
        rand_number = self.replay_online_rand_val[i % len(self.replay_online_rand_val)] 
        if self.comoda and rand_number < 0.5:
            data = self.get_comoda_data(i)
            replay_flag = True
        else: 
            # tot_samples = self.max_pretrain_frames + i 
            # online_data_thresh = i / tot_samples 
            online_data_thresh = 0.5
            if rand_number > online_data_thresh and self.apply_replay:
                data = self.get_replay_data(i)
                replay_flag = True 
            else:
                data = self.get_online_data(i)
                replay_flag = False
        return data, flag, replay_flag 
        '''if rand_number < 0.5 and self.apply_replay:
            data = self.get_replay_data(i)
            replay_flag = True  
        else:
            data = self.get_online_data(i)
            replay_flag = False 
        return data, flag, replay_flag'''  
    
    def get_data_flag(self, i):
        return self.OnlineDataset.data['flags'][i]

    def get_online_data(self, i):
        curr_frame = self.transforms(Image.open(self.OnlineDataset.data['curr_frames'][i]))
        next_frame = self.transforms(Image.open(self.OnlineDataset.data['next_frames'][i]))
        return {'curr_frame': curr_frame, 
                'next_frame': next_frame}

    def get_comoda_data(self, i):
        i_ = self.rand_pretrain_inds[i]
        curr_frame = self.transforms(Image.open(self.PretrainDataset.data['curr_frames'][i_]))
        next_frame = self.transforms(Image.open(self.PretrainDataset.data['next_frames'][i_]))
        return {'curr_frame': curr_frame, 
                'next_frame': next_frame}

    def get_replay_data(self, i):
        replay_curr_frames = [self.replay_curr_fr_dir + x for x in os.listdir(self.replay_curr_fr_dir) if x.endswith('.png')]
        replay_next_frames = [self.replay_next_fr_dir + x for x in os.listdir(self.replay_next_fr_dir) if x.endswith('.png')]
        u_rand = self.replay_pretrain_rand_val[i % len(self.replay_pretrain_rand_val)]
        tot_rep_samples = self.max_pretrain_frames + len(replay_curr_frames)
        # pretrain_replay_thresh = self.max_pretrain_frames / tot_rep_samples 
        pretrain_replay_thresh = 0.5
        if u_rand > pretrain_replay_thresh and len(replay_curr_frames) > 10:
            i_ = self.rand_replay_inds[i] % len(replay_curr_frames)
            curr_frame = self.transforms(Image.open(replay_curr_frames[i_]))
            next_frame = self.transforms(Image.open(replay_next_frames[i_]))
        else:
            i_ = self.rand_pretrain_inds[i]
            curr_frame = self.transforms(Image.open(self.PretrainDataset.data['curr_frames'][i_]))
            next_frame = self.transforms(Image.open(self.PretrainDataset.data['next_frames'][i_]))
            
        return {'curr_frame': curr_frame, 
                'next_frame': next_frame}
        
class KittiDepthTestDataset(): 
    def __init__(self, opts):
        self.opts = opts 
        self.test_dir = opts.kitti_test_in_dir 
        self.output_dir = opts.kitti_test_output_dir 
        self.gt_dir = opts.kitti_test_gt_dir
        self.list_transforms = []
        self.list_transforms.append(transforms.ToTensor())
        self.list_transforms.append(transforms.Normalize([0.45, 0.45, 0.45],
                                                         [0.225, 0.225, 0.225]))
        self.transforms = transforms.Compose(self.list_transforms)
        self.in_data = sorted([self.test_dir + x for x in os.listdir(self.test_dir) if x.endswith('.png')])
        self.gt_data = sorted([self.gt_dir + x for x in os.listdir(self.gt_dir) if x.endswith('.npy')])
        assert len(self.in_data) == len(self.gt_data), 'Different number of images to gt'
                
    def __len__(self):
        return len(self.in_data)
    
    def __getitem__(self, i):
        in_image = self.transforms(Image.open(self.in_data[i]))
        gt_depth = torch.tensor(np.load(self.gt_data[i]))
        return {'curr_frame': in_image,
                'gt': gt_depth}
        
class NYUDepthTestDataset(): 
    def __init__(self, opts):
        self.opts = opts 
        self.test_dir = opts.nyu_test_in_dir 
        self.output_dir = opts.nyu_test_output_dir 
        self.gt_dir = opts.nyu_test_gt_dir 
        self.list_transforms = []
        self.list_transforms.append(transforms.ToTensor())
        self.list_transforms.append(transforms.Normalize([0.45, 0.45, 0.45],
                                                         [0.225, 0.225, 0.225]))
        self.transforms = transforms.Compose(self.list_transforms)
        self.in_data = sorted([self.test_dir + x for x in os.listdir(self.test_dir) if x.endswith('.png')])
        self.gt_data = np.load(self.gt_dir + 'depth.npy')       
        assert len(self.in_data) == len(self.gt_data), 'Different number of images to gt'
                
    def __len__(self):
        return len(self.in_data)
    
    def __getitem__(self, i):
        in_image = self.transforms(Image.open(self.in_data[i]))
        gt = torch.tensor(self.gt_data[i])    
        return {'curr_frame': in_image,
                'gt': gt}    
        
class VirtualKittiDataset():
    def __init__(self, opts):
        self.opts = opts 
        self.in_dir = opts.v_kitti_root 
        self.vkitti_test_perc = opts.vkitti_test_perc 
        self.vkitti_exclude_domains = opts.vkitti_exclude_domains 
        scene_names = [self.in_dir + x + '/' for x in os.listdir(self.in_dir) if os.path.isdir(self.in_dir + x)]
        var_names = [x + '/' for x in os.listdir(scene_names[0]) if os.path.isdir(scene_names[0] + x)]
        for dom in self.vkitti_exclude_domains:
            var_names = [x for x in var_names if dom not in x]
        # gathering all the directories 
        full_seq_dir_names = []
        for scene_name in scene_names:
            for var_name in var_names:
                full_seq_dir_names.append(scene_name + var_name + 'frames/rgb/Camera_0/')
        full_seq_dir_names = sorted(full_seq_dir_names)
        
        # gathering all the frames 
        self.curr_frame_names = []
        self.next_frame_names = []
        for dir_name in full_seq_dir_names:
            seq_frame_names = [dir_name + x for x in os.listdir(dir_name) if x.endswith('.jpg')] 
            seq_frame_names = sorted(seq_frame_names)
            test_ind = int(len(seq_frame_names) * self.vkitti_test_perc)
            seq_frame_names = seq_frame_names[:-test_ind]
            self.curr_frame_names += seq_frame_names[:-1]
            self.next_frame_names += seq_frame_names[1:]
            
        self.data = {'curr_frames': self.curr_frame_names,
                        'next_frames': self.next_frame_names}
            
        assert len(self.curr_frame_names) == len(self.next_frame_names)
        
        self.list_transforms = []
        self.list_transforms.append(transforms.ToTensor())
        self.list_transforms.append(transforms.Normalize([0.45, 0.45, 0.45],
                                                         [0.225, 0.225, 0.225]))
        self.transforms = transforms.Compose(self.list_transforms)
        
    def __len__(self):
        return len(self.curr_frame_names)
    
    def __getitem__(self, i):
        curr_frame = self.transforms(Image.open(self.data['curr_frames'][i]))
        next_frame = self.transforms(Image.open(self.data['next_frames'][i]))
        return {'curr_frame': curr_frame, 
                'next_frame': next_frame}
        
        
class VirtualKittiWithKittiReplay():
    def __init__(self, opts):
        self.opts = opts 
        self.VirtualKitti = VirtualKittiDataset(self.opts)
        self.Kitti = KittiDataset(self.opts)
        
        self.replay_curr_fr_dir = self.opts.replay_curr_fr_dir
        self.replay_next_fr_dir = self.opts.replay_next_fr_dir
        self.pretrain_prob = self.opts.pretrain_prob 
        self.apply_replay = self.opts.apply_replay
        self.comoda = opts.comoda 

        self.replay_online_rand_val = np.random.rand(self.__len__())
        self.replay_pretrain_rand_val = np.random.rand(self.__len__())
        self.max_pretrain_frames = self.Kitti.__len__()
        self.max_replay_frames = 20000 
        self.rand_pretrain_inds = np.random.rand(self.__len__()) * (self.max_pretrain_frames - 1)
        self.rand_replay_inds = np.random.rand(self.__len__()) * (self.max_replay_frames - 1) 
        self.rand_pretrain_inds = self.rand_pretrain_inds.astype(int) 
        self.rand_replay_inds = self.rand_replay_inds.astype(int)
        
        self.list_transforms = []
        self.list_transforms.append(transforms.ToTensor())
        self.list_transforms.append(transforms.Normalize([0.45, 0.45, 0.45],
                                                         [0.225, 0.225, 0.225]))
        self.transforms = transforms.Compose(self.list_transforms)
        
        self.all_flags = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
        
    def __len__(self):
        return len(self.VirtualKitti)
    
    def __getitem__(self, i):
        flag = self.get_data_flag(i)
        rand_number = self.replay_online_rand_val[i % len(self.replay_online_rand_val)] 
        if rand_number < 0.5 and self.apply_replay:
            data = self.get_replay_data(i)
            replay_flag = True  
        else:
            data = self.get_online_data(i)
            replay_flag = False 
        return data, flag, replay_flag  
    
    def get_data_flag(self, i):
        curr_flag = [x for x in self.all_flags if x in self.VirtualKitti.data['curr_frames'][i]]
        # the above should be just a value 
        return curr_flag[0]
    
    def get_online_data(self, i):
        curr_frame = self.transforms(Image.open(self.VirtualKitti.data['curr_frames'][i]))
        next_frame = self.transforms(Image.open(self.VirtualKitti.data['next_frames'][i]))
        return {'curr_frame': curr_frame, 
                'next_frame': next_frame}

    def get_replay_data(self, i):
        if self.comoda:
            i_ = self.rand_pretrain_inds[i]
            curr_frame = self.transforms(Image.open(self.Kitti.data['curr_frames'][i_]))
            next_frame = self.transforms(Image.open(self.Kitti.data['next_frames'][i_])) 
        else:
            replay_curr_frames = [self.replay_curr_fr_dir + x for x in os.listdir(self.replay_curr_fr_dir) if x.endswith('.png')]
            replay_next_frames = [self.replay_next_fr_dir + x for x in os.listdir(self.replay_next_fr_dir) if x.endswith('.png')]
            u_rand = self.replay_pretrain_rand_val[i % len(self.replay_pretrain_rand_val)]
            if u_rand > 0.5 and len(replay_curr_frames) > 10:
                i_ = self.rand_replay_inds[i] % len(replay_curr_frames)
                curr_frame = self.transforms(Image.open(replay_curr_frames[i_]))
                next_frame = self.transforms(Image.open(replay_next_frames[i_]))
            else:
                i_ = self.rand_pretrain_inds[i]
                curr_frame = self.transforms(Image.open(self.Kitti.data['curr_frames'][i_]))
                next_frame = self.transforms(Image.open(self.Kitti.data['next_frames'][i_]))
            
        return {'curr_frame': curr_frame, 
                'next_frame': next_frame}        
    
class VirtualKittiDepthTestDataset():
    def __init__(self, opts):
        self.opts = opts 
        self.in_dir = opts.vkitti_test_in_dir
        self.gt_dir = opts.vkitti_test_gt_dir  
        self.vkitti_test_perc = opts.vkitti_test_perc 
        scene_names = [self.in_dir + x + '/' for x in os.listdir(self.in_dir) if os.path.isdir(self.in_dir + x)]
        d_scene_names = [self.gt_dir + x + '/' for x in os.listdir(self.gt_dir) if \
            os.path.isdir(self.gt_dir + x)]
        var_names = [x + '/' for x in os.listdir(scene_names[0]) if os.path.isdir(scene_names[0] + x)]
        # gathering all the directories 
        full_seq_dir_names = []
        full_depth_dir_names = []
        for scene_name in scene_names:
            for var_name in var_names:
                full_seq_dir_names.append(scene_name + var_name + 'frames/rgb/Camera_0/')
        for scene_name in d_scene_names:
            for var_name in var_names:
                full_depth_dir_names.append(scene_name + var_name + 'frames/depth/Camera_0/')
        full_seq_dir_names = sorted(full_seq_dir_names)
        full_depth_dir_names = sorted(full_depth_dir_names)
        # gathering all the frames 
        self.curr_frame_names = []
        self.gt_names = []
        for dir_name, d_dir_name in zip(full_seq_dir_names, full_depth_dir_names):
            seq_frame_names = [dir_name + x for x in os.listdir(dir_name) if x.endswith('.jpg')] 
            seq_frame_names = sorted(seq_frame_names)
            depth_frame_names = [d_dir_name + x for x in os.listdir(d_dir_name) if x.endswith('.png')]
            test_ind = int(len(seq_frame_names) * self.vkitti_test_perc)
            seq_frame_names = seq_frame_names[-test_ind:]
            depth_frame_names = depth_frame_names[-test_ind:]
            self.curr_frame_names += seq_frame_names
            self.gt_names += depth_frame_names 
            
        self.data = {'curr_frames': self.curr_frame_names,
                        'gt_frames': self.gt_names}
            
        assert len(self.curr_frame_names) == len(self.gt_names)
        
        self.list_transforms = []
        self.list_transforms.append(transforms.ToTensor())
        self.list_transforms.append(transforms.Normalize([0.45, 0.45, 0.45],
                                                         [0.225, 0.225, 0.225]))
        self.transforms = transforms.Compose(self.list_transforms)
        
    def __len__(self):
        return len(self.curr_frame_names)
    
    def __getitem__(self, i):
        curr_frame = self.transforms(Image.open(self.data['curr_frames'][i]))
        gt = torch.tensor(np.array(Image.open(self.data['gt_frames'][i])))
        return {'curr_frame': curr_frame, 
                'gt': gt}

        

if __name__ == '__main__':
    pretrain_opts = PreOptions().opts 
    online_opts = OnlineOptions().opts 
    test_opts = TestOptions().opts 
    
    NyuPre = NYUPretrain(pretrain_opts)
    NyuOn = NYUOnlineTrainDataset(online_opts)
    print(len(NyuPre))
    print(len(NyuOn))
    '''opts = Options().opts
    opts.v_kitti_dir = '/hdd/local/sdb/umar/vkitti_dataset/'
    VKitti = VirtualKittiDataset(opts)
    print(len(VKitti))
    data = VKitti.__getitem__(10)
    print(data['curr_frame'].size())
    print(data['next_frame'].size())
    print(data['intrinsics'].size())
    print(data['intrinsics_inv'].size())
    
    VKittiwKitti = VirtualKittiWithKittiReplay(opts)
    print(len(VKittiwKitti))
    print(len(VKittiwKitti.VirtualKitti))
    print(len(VKittiwKitti.Kitti))
    data, flag, replay_flag = VKittiwKitti.__getitem__(13500)
    print(VKittiwKitti.VirtualKitti.data['curr_frames'][13500])
    print(data['curr_frame'].size())
    print(data['next_frame'].size())
    print(data['intrinsics'].size())
    print(data['intrinsics_inv'].size())
    print(flag, replay_flag)'''
    


