import os
import torch
import torchvision.transforms as transforms
from PIL import Image

import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import WeightedRandomSampler


class GroupedDataset(Dataset):
   # for cifar4, should be 
    # means = [0.496887  0.4964557 0.4691896]
    # std = [0.23818445 0.23714599 0.25873092]
    
    # for isic (without sonic): should be 
    # means = [0.7519596 0.5541971 0.5523066]
    # std = [0.14961188 0.16191609 0.1755095 ]
    
    # for isic (with sonic - uncommon): should be 
    # means = [0.71835655 0.5626882  0.5254832]
    # std = [0.15869816 0.14007549 0.1677716]
    
    def __init__(self, 
                 data_dir, 
                 csv_fp, # relative to root dir
                 image_fp, # relative to root dir
                 split,
                 label_colname,
                 group_colnames=[], # list of column names we might later want to stratify on
                 this_group_key=None,  # str of the column name that serves as the subsetting group 
                                  # must be in group_colnames
                 transforms_basic = None,
                 transforms_augmented_subset = None,
                 image_colname = 'image_name'):
        
        group_id = this_group_key
        
#         print('root dir: ',data_dir)
#         print('image_fp: ',image_fp)
#         print(os.path.join(data_dir, image_fp))
        self.images_path = os.path.join(data_dir, image_fp)
#         print("images_path: ", self.images_path)
        
#        print('looking in ',os.path.join(data_dir,csv_fp))
         # self.full_data is a dataframe with labels, groups, and image file names
        self.full_data = pd.read_csv(os.path.join(data_dir,csv_fp))
        
        # get the df for this split (train/eval/test)
        split_df = self.full_data[self.full_data['fold'] == split]
        
        self.df = split_df.set_index(image_colname)
        self.split = split
        self.transforms_basic = transforms_basic
        
        if transforms_augmented_subset is not None:
            self.augment_subsets = True
            self.transforms_augmented_subset = transforms_augmented_subset
            
            augment_col = 'augment'
            assert augment_col in self.df.columns, '{0} must be a column of the dataframe'.format(augment_col)
            # if augmenting, make column for whether to agument
            # print(np.array(self.df[augment_col].values.sum()), 'aug')
            self.whether_to_augment = torch.from_numpy(np.array(self.df[augment_col].values.reshape(-1)))
        else:
            self.augment_subsets = False
        
        # track images for each entry
        image_names = []
        image_splits = []
        image_idxs = []
        
        for idx in range(len(self.df)):
            image_idxs.append(idx)
            image_splits.append(split)
            image_names.append(self.df.index[idx])
    
        self.image_names = image_names
        self.image_splits = image_splits
        self.image_idxs = image_idxs
        
        # instantiate targets
        self.targets = torch.from_numpy(np.array(self.df[label_colname].values))
        
        self.has_groups = (group_id is not None)
        
        group_info = {}
        # store the groups of each instance for downstream evaluation
        if group_colnames == [] and self.has_groups:
            print('instantiating group_colname as {0}'.format([group_id]))
            group_colnames = [group_id]
        
        
        for group_colname in group_colnames:
            group_info[group_colname] = torch.from_numpy(np.array(self.df[group_colname].values.reshape(-1)))
            
        self.group_info = group_info
        
        # instantiate groups
        self.groups = []
        self.group_counts = [] 
        
        # add groups     
        if self.has_groups:
            # the group on which to subset.
            self.group_id = group_id
            # unique values the groups can take on
            self.group_names = np.sort(np.unique(self.df[group_id].values))
            # 
            self.groups = group_info[group_id]
            
            self.group_counts = torch.zeros(len(self.group_names))
        
            for g, group_name in enumerate(self.group_names):
                self.group_counts[g] = (self.groups == group_name).sum()
        
               
        self.n = len(self.df)
        
    def __len__(self):
        return self.n
    
    
    def __getitem__(self, idx):
        
        # get image
        try: image = Image.open(os.path.join(self.images_path, self.image_names[idx]))
        except: 
            print('failed opening ', os.path.join(self.images_path, self.image_names[idx]))
   
        image = image.convert('RGB')

        if ((self.augment_subsets) and (self.whether_to_augment[idx])):
            image = self.transforms_augmented_subset(image)
       
        elif self.transforms_basic is not None:
            image = self.transforms_basic(image)
        
        image = np.array(image)
         
        # get sample    
        label = self.targets[idx]
        
        sample_this_index = {'image': image, 'target': label}
        
        for group_id_key in self.group_info.keys():
            sample_this_index[group_id_key] = self.group_info[group_id_key][idx]

        return sample_this_index

def get_transform(dataset_name, augment=False):
    
    channels = 3 
    resolution = 224 
    isic_no_sonic_normalization_stats = {'mean': (0.7519596, 0.5541971, 0.5523066), 'std': (0.14961188, 0.16191609, 0.1755095)}
    
    cifar_normalization_stats = {'mean': (0.496887, 0.4964557, 0.4691896), 'std': (0.23818445, 0.23714599, 0.25873092)} 

    isic_with_sonic_normalization_stats = {'mean': (0.71835655, 0.5626882, 0.5254832), 'std': (0.15869816, 0.14007549, 0.1677716)}
    
    if (dataset_name == 'isic'):
        normalization_stats = isic_no_sonic_normalization_stats
        print('using normalization data for ISIC without sonic')
    elif (dataset_name == 'cifar'):
        normalization_stats = cifar_normalization_stats
        print('using normalization data for CIFAR4')
    elif (dataset_name == 'isic_sonic'):
        normalization_stats = isic_with_sonic_normalization_stats
        print('using normalization data for ISIC with sonic')
    else:
        print('TODO: need to add the normalization stats for this dataset')
    
    test_transforms = [
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization_stats['mean'],
                             std=normalization_stats['std'])
    ]
    
    if augment:
        return transforms.Compose([transforms.RandomHorizontalFlip(),]
                                   #transforms.RandomVerticalFlip()]#,
                                 # transforms.RandomCrop(size=GroupedDataset._resolution)] 
                                   + test_transforms)
    else:
        return transforms.Compose(test_transforms)
        
def get_data_loaders(data_dir,
                     csv_fp,
                     image_fp,
                     label_colname,
                     eval_key,
                     dataset_name,
                     all_group_colnames = [],
                     this_group_key = None,
                     sample_by_groups = False,
                     weight_to_eval_set_distribution = True,
                     augment_subsets = False,
                     train_batch_size = 16, 
                     test_batch_size = 32,
                     num_workers = 32):
    
    # for now, don't augment the training set
    transform_basic = get_transform(dataset_name, augment=False)
    
    # train set will know to look for subsets to augment if transform_augment is not None
    if augment_subsets:
        print('augmenting')
        # can't do both right now
        assert not sample_by_groups, "can't augment and sample by groups rn" 
        transform_augment = get_transform(dataset_name, augment=True)
    else:
        transform_augment = None
        print('not augmenting')
    print()
    
    # get the datasets
    print('data dir: ',data_dir)
    print()
    train_set = GroupedDataset(data_dir, 
                                csv_fp,
                                image_fp,
                                split = 'train',  
                                label_colname = label_colname,
                                group_colnames = all_group_colnames,
                                this_group_key = this_group_key,
                                transforms_basic = transform_basic,
                                transforms_augmented_subset = transform_augment
                              )
    
    # don't do data augmentations for eval set of the traning set
    train_set_eval = GroupedDataset(data_dir, 
                            csv_fp,
                            image_fp,
                            split = 'train', 
                            label_colname = label_colname,
                            group_colnames = all_group_colnames,
                            this_group_key = this_group_key,
                            transforms_basic = transform_basic)
    
    val_set = GroupedDataset(data_dir, 
                          csv_fp,
                          image_fp,
                          split = 'val',  
                          label_colname = label_colname,
                          group_colnames = all_group_colnames,
                          this_group_key = this_group_key,
                          transforms_basic = transform_basic)
    
    test_set = GroupedDataset(data_dir, 
                           csv_fp,
                           image_fp,
                           split = 'test', 
                           label_colname = label_colname,
                           group_colnames = all_group_colnames,
                           this_group_key = this_group_key,
                           transforms_basic = transform_basic)
    
    
    
    if not this_group_key is None:
        print('group_names: ',train_set.group_names)
        print()
        
    group_dict = {}
    
    # sample by groups is importance sampling
    if sample_by_groups:
        assert not this_group_key is None
        shuffle_train = False
        
        # confirm whether weighting by group is what you actually want in all cases,
        # or modify code to handle other cases -- could definitely weight 
        # by test set proportions, as the following code will do
        # can also specify the weights in an input variable
        print('training set group counts:',train_set.group_counts)        
        if weight_to_eval_set_distribution:
            if (eval_key == 'test'): # we use this case for the subsetting experiment
                print("using TEST to set weights")
                # print('test (eval) set group counts ',test_set.group_counts)
                test_fracs = test_set.group_counts  / test_set.group_counts.sum() 
            elif (eval_key == 'val'): # we use this case for the HP search experiment
                print("using VAL to set weights")
                # print('val (eval) set group counts ',val_set.group_counts)
                test_fracs = val_set.group_counts  / val_set.group_counts.sum()

            train_fracs = train_set.group_counts / train_set.group_counts.sum() 
            weights_by_group = test_fracs / train_fracs 
          
        else:
            # if running GDRO, e.g., we want each group to be sampled equally, not
            # according to their test set percentages
            weights_by_group = train_set.group_counts.sum() / train_set.group_counts
        
        group_labels_by_instance = train_set.groups 
        instance_weights = weights_by_group[group_labels_by_instance].type(torch.DoubleTensor)
        
        print('group training weights:',weights_by_group)
        print()
       # print('confirming group weights ', [instance_weights[group_labels_by_instance == g].mean() for g in [0,1]])
      #  print(instance_weights.dtype)
        train_sampler = WeightedRandomSampler(instance_weights, 
                                              len(group_labels_by_instance), 
                                              replacement=True)
        
        group_dict['group_counts_train'] = train_set.group_counts
        group_dict['num_groups'] = len(weights_by_group)
    
#     elif augment_subsets:
#         shuffle_train = False
        
#         # double weight the samples to be augmented
#         to_augment_by_instance = train_set.whether_to_augment 
#     #    print(len(to_augment_by_instance))
#         print('augmenting a total of {0} instances'.format(to_augment_by_instance.sum()))
#         instance_weights = np.ones(len(to_augment_by_instance))
#         instance_weights[to_augment_by_instance] += 1
#       #  instance_weights = instance_weights / instance_weights.sum()
#         print('instance weights for augmented sample: ',np.unique(instance_weights[to_augment_by_instance]))
#         print('instance weights for not-augmented samples: ',np.unique(instance_weights[~to_augment_by_instance]))
#         print('percent of augmented samples from group 0 :', 1-(train_set.groups[to_augment_by_instance]).numpy().mean())
        
#         train_sampler = WeightedRandomSampler(instance_weights, 
#                                               len(to_augment_by_instance), 
#                                               replacement=True)
            
    else:
        shuffle_train = True
        train_sampler = None
        
    print('train batch size', train_batch_size)
    train_loader = DataLoader(train_set, 
                              batch_size = train_batch_size,
                              shuffle = shuffle_train,
                              sampler = train_sampler,
                              num_workers = num_workers)
    
    train_loader_eval = DataLoader(train_set, 
                              batch_size = train_batch_size,
                              shuffle = False,
                              num_workers = num_workers)
    
    val_loader = DataLoader(val_set, 
                            batch_size = train_batch_size,
                            shuffle = False,
                            num_workers = num_workers)
    
    test_loader = DataLoader(test_set, 
                             batch_size = train_batch_size,
                             shuffle = False,
                             num_workers = num_workers)
    
    
    return train_loader, train_loader_eval, val_loader, test_loader, group_dict
       
    
    
    
    
    
        
