import os 
# from options import Options 
import numpy as np 


class NYUCategorySplit():
    def __init__(self, opts):
        self.opts = opts 
        self.nyu_root = self.opts.nyu_root 
        self.nyu_cat_file_name = self.opts.nyu_cat_file_name

    def __call__(self, overwrite=False):
        if overwrite or not os.path.exists(self.nyu_cat_file_name): 
            self.seq_names = [x for x in os.listdir(self.nyu_root) if os.path.isdir(self.nyu_root + x)]
            self.seq_categories = self.get_categories()
            self.all_categories = self.gather_category_frames()
            self.category_name_dict = self.make_dictionary()
            np.save(self.nyu_cat_file_name, self.category_name_dict)
        else:
            self.category_name_dict = np.load(self.nyu_cat_file_name, allow_pickle=True).item()
        self.num_of_sequences = 0 
        for cat_name, cat_sequences in self.category_name_dict.items():
            self.num_of_sequences += len(cat_sequences)
        return self.category_name_dict 
        
    def get_categories(self): 
        seq_categories = [x.split('0')[0] for x in self.seq_names]
        seq_categories = sorted(list(set(seq_categories)))  # getting rid of repitition 
        return seq_categories

    def gather_category_frames(self):
        all_categories = [] 
        for seq_category in self.seq_categories:
            seqs = sorted([x for x in self.seq_names if x.startswith(seq_category)]) 
            all_categories.append(seqs) 
        return all_categories 

    def make_dictionary(self):
        assert len(self.seq_categories) == len(self.all_categories), 'different number of categories to list'
        nyu_cat_dictionary = {}
        for cat_name, cat_sequences in zip(self.seq_categories, self.all_categories):
            nyu_cat_dictionary[cat_name] = cat_sequences
        return nyu_cat_dictionary 



if __name__ == '__main__':
    catSplit = NYUCategorySplit(Options().opts)
    name_dict = catSplit(overwrite=True) 
    print('Categories: {}'.format(catSplit.seq_categories))
    print('Total categories: {}'.format(catSplit.num_of_categories))
    print('Total sequences: {}'.format(catSplit.num_of_sequences))



    
        
        