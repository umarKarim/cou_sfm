# Towards Continual, Online, Unsupervised Depth  
## Introduction 
This is the source code for the paper **Towards Continual, Online, Unsupervised Depth**. This code is for Structure from Motion (SfM)-based depth estimation. The stereo-based depth estimation is also available. 

## Requirements 
- PyTorch 
- Torchvision 
- NumPy 
- Matplotlib 
- OpenCV
- Torchvision
- Pandas 

# KITTI-NYU Experiments 
 ## Data Preparation
 Download the [KITTI](https://1drv.ms/u/s!AiV6XqkxJHE2g1zyXt4mCKNbpdiw?e=ZJAhIl) dataset, the rectified [NYU](https://drive.google.com/file/d/1oy5TpkMusgyJQpkXpqfQI7rvLt_RK5En/view?usp=sharing), the KITTI test [dataset](https://1drv.ms/u/s!AiV6XqkxJHE2kz5Zy7jWZd2GyMR2?e=kBD4lb) and the NYU test [dataset](https://1drv.ms/u/s!AiV6XqkxJHE2kz85ZcYiCoZmSjKk?e=qGpvck). Extract data to appropriate locations. Saving SSD is encouraged but not required. Virtual KITTI is not required for the KITTI-NYU experiments. Similarly, NYU is not required for KITTI-vKITTI experiments. The directory names are slightly changed (by adding an underscore) for the NYU dataset for categorization of the code.

## Pre-Training 
Set paths in the *dir_options/pretrain_options.py* file. Then run 

```
python pretrain.py
```
The pre-trained models should be saved in the directory *trained_models/pretrained_models/*.

## Online Training 
Set paths in the *dir_options/online_train_options.py* file. Then run 

```
python script_online_train.py
```
The online-trained models (for a single epoch only) will be saved in the *trained_models* directory. Intermediate results will be saved in the *qual_dmaps* directory. 

## Testing 
Set paths in *dir_options/test_options.py* file. To determine the categories of test data, first change the paths in the *dir_dataset/test_data_categorize.py* and run

```
python test_data_categorize.py
```
Follow up by

```
python script_test_directory.py
```

Results will be stored in the *results* directory. Then run 

``` 
python script_evaluate.py
```

Results will be displayed in the console.

## Results 
Quantitative results are mentioned in the manuscript. Check the following [video](https://www.youtube.com/watch?v=_WNYOTDaCCM&t=10s&ab_channel=Depth) for qualitative results. 


# KITTI-vKITTI Experiments 
 ## Data Preparation
 Download the [KITTI](https://1drv.ms/u/s!AiV6XqkxJHE2g1zyXt4mCKNbpdiw?e=ZJAhIl) dataset, the Virtual KITTI [RGB](http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar), and the KITTI test [dataset](https://1drv.ms/u/s!AiV6XqkxJHE2kz5Zy7jWZd2GyMR2?e=kBD4lb).Extract data to appropriate locations. Saving SSD is encouraged but not required. Virtual KITTI is not required for the KITTI-NYU experiments. Similarly, NYU is not required for KITTI-vKITTI experiments. 

## Pre-Training
Set the paths in *dir_options/pretrain_options.py*. Then run the following

~~~
python script_kitti_pretrain.py
~~~

## Online Traning 
Set the paths in *dir_options/online_train_options.py*. Then run the following 

~~~
python script_vkitti_exp.py
~~~

## Evaluation
Set the paths in *dir_options/test_options.py*. Then run 

~~~
python script_test_vkitti_exp.py
~~~








