import numpy as np 
import os 
# from test_options import Options 

class TestCatNames(): 
    def __init__(self, opts):
        self.kitti_eigen_test_split_file = opts.kitti_eigen_test_split_file 
        self.kitti_categories = ['city', 'city', 'city', 'residential', 'residential', 'road', 'road', 'residential', 
             'residential', 'city', 'road', 'city', 'city', 'residential', 'city', 'residential', 'city',
             'city', 'road', 'city', 'city', 'city', 'city', 'road', 'residential',
             'residential', 'residential', 'road']
        self.kitti_total_frames = 696 
        self.kitti_test_cat_file_name = opts.kitti_test_cat_file_name 
        
        self.nyu_text_file_name = opts.nyu_text_file_name 
        self.nyu_test_in_dir = opts.nyu_test_in_dir 
        self.nyu_test_cat_file_name = opts.nyu_test_cat_file_name 
        
        self.test_categorize_kitti()
        print('Names of kitti test categories saved')
        self.test_categorize_nyu()
        print('Names of nyu test categories saved')
    
    def test_categorize_kitti(self):
        dir_names = []
        dir_ind = []
        with open(self.kitti_eigen_test_split_file, 'r') as f:
            tot_frames = 0
            prev_dir_name = 'abc'
            for ind, im_info in enumerate(f):
                tot_frames += 1
                full_name = im_info.split('\n')[0]
                dir_name = (full_name.split('/')[1]).split(' ')[0]
                if dir_name != prev_dir_name:
                    dir_names += [dir_name]
                    dir_ind += [ind]
                    prev_dir_name = dir_name
        cat_set = sorted(list(set(self.kitti_categories)))
        
        cat_len = len(self.kitti_categories)
        dir_name_len = len(dir_names)
        dir_ind_len = len(dir_ind)
        assert cat_len == dir_name_len == dir_ind_len, 'Check again, {} vs {} vs {}'.format(cat_len, 
                                                                                            dir_name_len,
                                                                                            dir_ind_len)
        dict = {}
        for key in cat_set:
            dict[key] = []
        for i, st_pt in enumerate(dir_ind):
            if i == len(dir_ind) - 1:
                curr_range = list(range(st_pt, tot_frames)) 
            else:
                curr_range = list(range(st_pt, dir_ind[i+1]))
            curr_cat = self.kitti_categories[i]
            dict[curr_cat] += curr_range 
        print('KITTI')
        print(dict)
        np.save(self.kitti_test_cat_file_name, dict)

    def test_categorize_nyu(self):
        f = open(self.nyu_text_file_name)
        scenes = []
        for line in f:
            line_ = line.split('/')
            curr_line = line.split('/')[0]
            scene = curr_line.split('_0')[0]
            scenes += [scene]

        input_names = [x for x in os.listdir(self.nyu_test_in_dir) if x.endswith('.png')]
        input_names = sorted(input_names)
        input_ind = [int(x.split('.')[0]) - 1 for x in input_names]
        
        scenes = np.array(scenes)
        input_ind = np.array(input_ind)
        corr_scenes = scenes[input_ind]
        normal_range = np.arange(0, len(corr_scenes), 1)
        
        keys = sorted(list(set(corr_scenes)))
        data_dict = {}
        for key in keys:
            mask = corr_scenes == key 
            vals = normal_range[mask] 
            data_dict[key] = vals
        print('NYU')
        print(data_dict) 
        np.save(self.nyu_test_cat_file_name, data_dict)


if __name__ == '__main__':
    opts = Options().opts
    opts.kitti_eigen_test_split_file = '/hdd/local/sdb/umar/codes/continual_sfm/dir_filenames/kitti_eigen_test_split.txt'
    opts.nyu_text_file_name = '/hdd/local/sdb/umar/codes/continual_sfm/dir_filenames/nyu_text_file.txt'
    TestCatNames(opts)