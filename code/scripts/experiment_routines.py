import sys
import train_model
import eval_results

import torch
import torch.optim as optim
import torch.nn as nn
import pickle
import os
import pandas as pd
import numpy as np

# local imports
from dataset import get_data_loaders 
from dataset_chunking_fxns import subsample_df_by_groups

DATA_DIR = '../../data'

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return
    

def train_and_eval_basic(csv_fp,
                         image_fp,
                         label_colname,
                         eval_key,
                         dataset_name,
                         group_key=None,
                         augment_subsets = False,
                         all_group_colnames=[],
                         data_dir=DATA_DIR,
                         num_workers=64,
                         seed = None,
                         num_classes=2, 
                         num_epochs = 20,
                         return_model = False,
                         model_name = 'resnet18',
                         **sgd_params):
    print("Training and evaluating ERM") 
    # set up the dataloaders 
    dataloaders = get_data_loaders(data_dir = data_dir,
                                   csv_fp = csv_fp,
                                   image_fp = image_fp,
                                   label_colname = label_colname,
                                   eval_key = eval_key,
                                   dataset_name = dataset_name,
                                   all_group_colnames = all_group_colnames,
                                   this_group_key = group_key,
                                   augment_subsets = augment_subsets,
                                   sample_by_groups = False,
                                   num_workers=num_workers)
    
    x = train_and_eval(dataloaders,
                       eval_key,
                       seed,
                       num_classes, 
                       num_epochs,
                       return_model,
                       model_name,
                       **sgd_params)
        
    return x

def train_and_eval_IS(csv_fp,
                      image_fp,
                      label_colname,
                      eval_key,
                      dataset_name,
                      group_key,
                      all_group_colnames=[],
                      data_dir=DATA_DIR,
                      num_workers=64,
                      seed = None,
                      num_classes=2, 
                      num_epochs = 20,
                      return_model = False,
                      model_name = 'resnet18',
                      **sgd_params):
   
    print("Training and evaluating IS")
    # set up the dataloaders 
    dataloaders = get_data_loaders(data_dir = data_dir,
                                   csv_fp = csv_fp,
                                   image_fp = image_fp,
                                   label_colname = label_colname,
                                   eval_key = eval_key,
                                   dataset_name = dataset_name,
                                   all_group_colnames = all_group_colnames,
                                   this_group_key = group_key,
                                   sample_by_groups = True,
                                   weight_to_eval_set_distribution = True,
                                   num_workers=num_workers)
    
    x = train_and_eval(dataloaders,
                       eval_key,
                       seed,
                       num_classes, 
                       num_epochs,
                       return_model,
                       model_name,
                       **sgd_params)
        
    return x

def train_and_eval_GDRO(csv_fp,
                        image_fp,
                        label_colname,
                        eval_key,
                        dataset_name,
                        group_key,
                        gdro_params,
                        all_group_colnames=[],
                        data_dir=DATA_DIR,
                        num_workers=64,
                        seed = None,
                        num_classes=2, 
                        num_epochs = 20,
                        return_model = False,
                        model_name = 'resnet18',
                        **sgd_params):
    print("Training and Evaluating GDRO")
    # set up the dataloaders - GDRO needs sample_by_groups = True
    dataloaders = get_data_loaders(data_dir = data_dir,
                                   csv_fp = csv_fp,
                                   image_fp = image_fp,
                                   label_colname = label_colname,
                                   eval_key=eval_key,
                                   dataset_name=dataset_name,
                                   all_group_colnames = all_group_colnames,
                                   this_group_key = group_key,
                                   sample_by_groups = True,
                                   weight_to_eval_set_distribution = False,
                                   num_workers=num_workers)
    
    # update gdro params with dataset specific paramters
    dataloader_train = dataloaders[0]
    gdro_params['group_key'] = group_key
    gdro_params['num_groups'] = len(dataloader_train.dataset.group_counts)
    gdro_params['group_sizes'] = dataloader_train.dataset.group_counts

    x = train_and_eval(dataloaders,
                       eval_key,
                       seed,
                       num_classes, 
                       num_epochs,
                       return_model,
                       model_name,
                       gdro=True,
                       gdro_params=gdro_params,
                       **sgd_params)
        
    return x
        
        
  
def train_and_eval(dataloaders,
                   eval_key,
                   seed = None,
                   num_classes=2, 
                   num_epochs = 20,
                   return_model = False,
                   model_name='resnet18',
                   gdro=False,
                   gdro_params={},
                   **sgd_params,
                   ):
    
    # set seed
    if not seed is None: set_seeds(seed)
    
    # new: dataloaders should already determing if there's IS or now
    train_loader, train_eval_loader, val_loader, test_loader, _ = dataloaders
    
    dataloaders = {'train': train_loader, 
                   'val': val_loader,
                   'test': test_loader}
    
    print('Using', model_name)
    model, input_size = train_model.initialize_model(model_name, num_classes)

    params_to_update = model.parameters()
    
    print()
    print("SGD parameters:", sgd_params)
    optimizer = optim.SGD(params_to_update, 
                          lr = sgd_params['lr'], 
                          weight_decay = sgd_params['weight_decay'], 
                          momentum = sgd_params['momentum'])
    
    if gdro:
        print()
        print("GDRO Parameters:", gdro_params)
         # don't reduce because we need to take a weighted sum of losses
        criterion_train = nn.CrossEntropyLoss(reduction='none')
    else:
        criterion_train = nn.CrossEntropyLoss()
    print()
    
    print('train loader is of size', len(train_loader.dataset))
    
    train_model.train_net(model, 
                          dataloaders, 
                          criterion_train,  
                          optimizer,  
                          gdro=gdro,
                          num_epochs=num_epochs,
                          **gdro_params)
    
    criterion_eval = nn.CrossEntropyLoss()
    print("evaluating performance on", eval_key)
    preds, labels = train_model.eval_model(model, 
                                           dataloaders, 
                                           criterion_eval, 
                                           eval_set = eval_key, 
                                           return_preds = True)

    if return_model:
        return preds, dataloaders, model
    else:
        return preds, dataloaders


def subset_and_train(csv_fp,
                     image_fp,
                     label_colname,
                     all_group_colnames,
                     group_key, 
                     subset_groups, 
                     subset_sizes, 
                     train_fxn, 
                     sgd_params,
                     save_fn,
                     eval_key,
                     dataset_name,
                     adjust_epochs_according_to_n = False,
                     train_fxn_params = {},
                     gdro=False,
                     gdro_params = {},
                     replace=False,
                     seed = None,
                     data_dir=DATA_DIR,
                     tmp_data_descriptor='debug',
                     model_name = 'resnet18',
                     return_model = False,
                     use_diff_wds=False,
                     wd_range=[-1,sys.maxsize]):
    ''' 
    group_key (str): the column of the dataframe on which to define the subset
    subset_groups (list of g lists): the groups that make up the subsets
    subset_frac (k x g ndarray): where k is the number of fractional 
        settings to search over, and g the number of subset groups. 
        Fractions are defined wrt the total size of each subgroup, not the 
        entire training set.
    '''
    
    int_data_dir = 'int_datasets_subsetted'
    if not os.path.exists(os.path.join(data_dir,int_data_dir,tmp_data_descriptor)):
        os.makedirs(os.path.join(data_dir,int_data_dir,tmp_data_descriptor))
    
    
    print('looking for csv in ', csv_fp)
    full_data = pd.read_csv(os.path.join(data_dir, csv_fp)).copy().reset_index(drop=True)
    
    
    rs = np.random.RandomState(seed)
    
    results_dict = {}
    results_dict['group_key'] = group_key
    results_dict['subset_groups'] = subset_groups
    results_dict['subset_sizes'] = subset_sizes
    results_dict['accs_by_group'] = []
    results_dict['accs_total'] = []
    
    if (use_diff_wds):
        diff_wds = sgd_params['weight_decay'].copy()
            
    for i, group_sizes_this_trial in enumerate(subset_sizes):
        print()
        print('on subset ', i+1, 'out of', len(subset_sizes))
        
        if (use_diff_wds):
            if (group_sizes_this_trial[0] <= wd_range[0] or group_sizes_this_trial[0] >= wd_range[1]):
                sgd_params['weight_decay'] = diff_wds[0]
            else:
                sgd_params['weight_decay'] = diff_wds[1]
        
        save_fn_tmp = 'tmp_dataset_{0}.csv'.format(i)
       
        # construct a new dataframe with the subsampled groups
        df_sampled_this_trial = subsample_df_by_groups(full_data, 
                                                       group_key,
                                                       subset_groups, 
                                                       group_sizes_this_trial, 
                                                       rs,
                                                       keep_test_val=True)
        
        # shuffle
        df_sampled_this_trial = df_sampled_this_trial.sample(frac=1, 
                                                             replace=False,
                                                             random_state = rs).reset_index(drop=True)
        
        # names for the tmp dataset
        int_csv_for_data_loader = os.path.join(int_data_dir,
                                               tmp_data_descriptor,
                                               save_fn_tmp)
        
        int_csv_fp_to = os.path.join(data_dir, 
                                     int_csv_for_data_loader)
        
        print("filepath to tmp data CSV", int_csv_fp_to)
        print()
        
        df_sampled_this_trial.to_csv(int_csv_fp_to)
        
        # if adjusting for different n, change the num_epcohs in the training fxn params
        train_fxn_params_new = train_fxn_params.copy()
        
        if adjust_epochs_according_to_n:
            n_orig = train_fxn_params['base_train_size_for_num_epochs']
            n_this = (df_sampled_this_trial['fold'] == 'train').sum()
            num_epochs_base = train_fxn_params['num_epochs']
            num_epochs_adj = int(np.round(num_epochs_base *n_orig / n_this))
            print('(before) with {0} samples, we did {1} epochs'.format(n_orig, num_epochs_base))
            print('with {0} training samples, doing {1} epochs'.format(n_this, num_epochs_adj))
            train_fxn_params_new['num_epochs'] = num_epochs_adj
        
        if gdro:
            preds, dataloaders = train_fxn(int_csv_for_data_loader,
                                           image_fp,
                                           label_colname,
                                           eval_key,
                                           dataset_name,
                                           group_key,
                                           gdro_params,
                                           all_group_colnames = all_group_colnames,
                                           data_dir=DATA_DIR,
                                           seed=seed, 
                                           model_name = model_name,
                                           **train_fxn_params_new, 
                                           **sgd_params)
            
        else:
            preds, dataloaders = train_fxn(int_csv_for_data_loader,
                                           image_fp,
                                           label_colname,
                                           eval_key,
                                           dataset_name,
                                           group_key,
                                           all_group_colnames = all_group_colnames,
                                           data_dir=DATA_DIR,
                                           seed=seed, 
                                           model_name = model_name,
                                           **train_fxn_params_new, 
                                           **sgd_params)
        
        data_eval_this = dataloaders[eval_key]
        
        accs = eval_results.grab_results_by_group(data_eval_this,
                                                  preds, 
                                                  group_key = group_key)
        
        accs_by_group, accs_total = accs
        results_dict['accs_by_group'].append(accs_by_group)
        results_dict['accs_total'].append(accs_total)

   
    print('saving results in ',save_fn)
    with open(save_fn, 'wb') as f:
        pickle.dump(results_dict, f)
    
    return 
