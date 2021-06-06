import numpy as np

#import seaborn as sns
import matplotlib
import torch
from importlib import reload
from PIL import Image
import pandas as pd

import os
import sys
# use relative imports
project_root_dir = '../../'
sys.path.append(os.path.join(project_root_dir, 'code'))
from make_subset_batches import subset_set, subset_labels, subset_classes

#local import from res_net folder
sys.path.append(os.path.join(project_root_dir,'res_net'))
#from core import *
from torch_backend import cifar10



print("keeping only entries from groups {0}".format(subset_classes))
DATA_DIR = './data'
dataset_10 = cifar10(DATA_DIR)
dataset_4 = {}
for t in ['train', 'valid']:
    dataset_4[t] = subset_set(dataset_10[t], subset_labels)

n = len(dataset_4['valid']['targets'])
m = len(dataset_4['train']['targets'])

bird_map = {0:0, 1:0, 2:1, 7:0}
bird1 = [bird_map[dataset_4['valid']['targets'][i]] for i in range(n)]
bird2 = [bird_map[dataset_4['train']['targets'][i]] for i in range(m)]
bird = bird1 + bird2

airplane_map = {0:1, 1:0, 2:0, 7:0}
airplane1 = [airplane_map[dataset_4['valid']['targets'][i]] for i in range(n)]
airplane2 = [airplane_map[dataset_4['train']['targets'][i]] for i in range(m)]
airplane = airplane1 + airplane2

horse_map = {0:0, 1:0, 2:0, 7:1}
horse1 = [horse_map[dataset_4['valid']['targets'][i]] for i in range(n)]
horse2 = [horse_map[dataset_4['train']['targets'][i]] for i in range(m)]
horse = horse1 + horse2

automobile_map = {0:0, 1:1, 2:0, 7:0}
automobile1 = [automobile_map[dataset_4['valid']['targets'][i]] for i in range(n)]
automobile2 = [automobile_map[dataset_4['train']['targets'][i]] for i in range(m)]
automobile = automobile1 + automobile2

animal_map = {0:0, 1:0, 2:1, 7:1}
animal1 = [animal_map[dataset_4['valid']['targets'][i]] for i in range(n)]
animal2 = [animal_map[dataset_4['train']['targets'][i]] for i in range(m)]
animal = animal1 + animal2

air_map = {0:1, 1:0, 2:1, 7:0}
air1 = [air_map[dataset_4['valid']['targets'][i]] for i in range(n)]
air2 = [air_map[dataset_4['train']['targets'][i]] for i in range(m)]
air = air1 + air2

fold_original = ['test']*n + ['train']*m 

fold1 = ['train' for i in range((n+m)//3)]
fold2 = ['val' for i in range((n+m)//3)]
fold3 = ['test' for i in range(n+m - 2*(n+m)//3)]
fold = fold1 + fold2 + fold3
np.random.shuffle(fold)

image_name = [str(i)+".jpeg" for i in range(m+n)]
# intialise data of lists. 
data = {'bird':bird, 'airplane':airplane, 'horse':horse, 'automobile':automobile, 
        'animal':animal, 'air':air, 'image_name':image_name, 'fold':fold_original, 'fold_new':fold} 
  
# Create DataFrame 
labels_df = pd.DataFrame(data) 

project_data_dir = '../../data'

out_dir_data = os.path.join(project_data_dir,'cifar4')

out_dir_image = os.path.join(project_data_dir, 'cifar4/images')

#if out dirs dont exist, make them
for out_dir in [out_dir_data, out_dir_image]:
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
labels_df.to_csv(os.path.join(out_dir_data, 'df_cifar4_labels.csv'), index=False)



for i in range(n):
    file_name = os.path.join(out_dir_image, str(i)+".jpeg")
    im = Image.fromarray(dataset_4['valid']['data'][i])
    im.save(file_name)
    
for i in range(m):
    file_name = os.path.join(out_dir_image, str(n+i)+".jpeg")
    im = Image.fromarray(dataset_4['train']['data'][i])
    im.save(file_name)
