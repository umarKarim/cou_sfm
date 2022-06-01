import numpy as np

dict_path = 'results/sfml/online_test_loss/nyu_online/nyutrain_nyutest_online.npy'
metric = 'abs_rel'
data = np.load(dict_path, allow_pickle=True).item()
print(data[metric][-1]['cat'])