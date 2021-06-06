import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold

def add_stratified_test_splits(original_dataframe,
                               group_key,
                               test_frac = 0.2):
    
    rs = np.random.RandomState(0)
    data = original_dataframe.copy()
    
    # default to train
    data.loc[:,'fold'] = 'train'
    
    # for each group, put test_frac fraction of point as test
    for group_id in np.sort(data[group_key].unique()):
        # indexes into the keys that are this group
        idxs_this_group = np.where(data[group_key] == group_id)[0]
        # pick a subset of these to be test instances
        random_idxs_this_group = rs.choice(len(idxs_this_group), 
                                           int(test_frac * len(idxs_this_group)),
                                           replace=False)
        # index into original dataframe
        test_idxs_this_group = idxs_this_group[random_idxs_this_group]
        data.loc[test_idxs_this_group,'fold'] = 'test'
        
    # return new dataframe
    return data

def add_stratified_kfold_splits(original_dataframe_fp, 
                                group_keys,
                                num_splits=5,
                                overwrite=False
                                ):
    
    out_csv_fp = original_dataframe_fp.replace('.csv','_{0}_{1}_fold_splits.csv'.format(num_splits, group_keys))
    
    write_file = True
    if os.path.exists(out_csv_fp) and not overwrite:
        print(out_csv_fp, ' exists, wont overwrite without overwrite=True (will return the data)')
        write_file=False
    
    data = pd.read_csv(original_dataframe_fp)
    
    kf_strat = StratifiedKFold(num_splits, shuffle=True, random_state=0)    
    
    train_data_idxs = data.loc[data['fold'] == 'train'].index
    test_data_idxs = data.loc[data['fold'] == 'test'].index
    train_data = data.loc[train_data_idxs]
    
    # make sure the array is one-hot encoded or one_dimensional
    if len(train_data[group_keys].shape) > 1 and train_data[group_keys].shape[-1] != 1: 
        print('assuming data with the group format is one-hot encoded')
        # make sure groups are  all 0's or 1's
        assert np.array_equal(train_data[group_keys], train_data[group_keys].astype(bool))
        # if one-hot encoded (binary and not 1d), make a multiclass label
        train_groups = train_data[group_keys].values.dot(np.arange(train_data[group_keys].values.shape[1]))

    else:
        train_groups = train_data[group_keys].values
    
    # this kf will stratify on the second argument
    splits = kf_strat.split(train_data_idxs, 
                            train_groups)
    
    for f, (train_idxs, val_idxs) in enumerate(splits):
        this_fold_key = 'cv_fold_{0}'.format(f)
        data.loc[:,this_fold_key] = 'test'

        this_split_train_idxs = train_data_idxs[train_idxs]
        this_split_val_idxs = train_data_idxs[val_idxs]
        data.loc[this_split_train_idxs, this_fold_key] = 'train'
        data.loc[this_split_val_idxs, this_fold_key] = 'val'
        
    # some quick checks
    cv_folds = ['cv_fold_{0}'.format(x) for x in range(num_splits)]
    # check that every training point got assigned to a exactly one validation set
    assert ((data[cv_folds] == 'val').sum(axis=1) ==1)[train_data_idxs].all()
    # check that all test idxs remain test idxs
    assert np.array(data.loc[test_data_idxs, cv_folds] == 'test').all()
    # check that for each of the groups, there's no more than one difference in each validation split
    for g, group_key in enumerate(np.unique(train_groups)):
        num_vals_per_group = (data.loc[train_data_idxs,cv_folds][train_groups == g] == 'val').sum().values
        assert num_vals_per_group.max() - num_vals_per_group.min() <= 1, \
                        'group {0} fold counts: {1}'.format(group_key,num_vals_per_group)
    
    
    # save output in a new csv in the same folder as the input csv
    if write_file:
        print('writing split file in ', out_csv_fp)
        data.to_csv(out_csv_fp, index=False)

    return data


def get_single_cv_kfold_split(kfold_csv_fp,
                              split_seed,
                              save_fn = None):
         
        
    full_data = pd.read_csv(kfold_csv_fp).copy()
    
    this_split_fold_key = 'cv_fold_{0}'.format(split_seed)
    
    # make a copy
    data_1 = full_data.copy()
    # rename to train and val
    data_1.loc[:,'fold'] = data_1.loc[:,this_split_fold_key]
    
    if save_fn is not None:
        # print('saving tmp csv in ',save_fn)
        data_1.to_csv(save_fn, index=False)
        
    return data_1

def add_split_row(full_data, dataset_split_seed, splits):
    train1_pct,train2_pct, val1_pct, val2_pct  = splits

    n_full = len(full_data)
    n_train1 = int(train1_pct * n_full)
    n_train2 = int(train2_pct * n_full)
    n_val1 = int(val1_pct * n_full)
    n_val2 = n_full - (n_train1 + n_train2 + n_val1)

    # reset random seed
    rs = np.random.RandomState(dataset_split_seed)
    shuffled_idxs = rs.choice(n_full, n_full, replace=False)
    
    idxs_train1 = shuffled_idxs[:n_train1]
    idxs_train2 = shuffled_idxs[n_train1:n_train1+n_train2]
    idxs_val1 = shuffled_idxs[n_train1+n_train2: n_train1+n_train2+n_val1]
    idxs_val2 = shuffled_idxs[n_train1 + n_train2 + n_val1 :n_train1+n_train2+n_val1 + n_val2]


    # instantiate the splits
    fold_split_key = 'fold_split_{0}'.format(dataset_split_seed)
    full_data[fold_split_key] = ''

    full_data.loc[shuffled_idxs[idxs_train1],fold_split_key] = 'train_1'
    full_data.loc[shuffled_idxs[idxs_train2],fold_split_key] = 'train_2'

    full_data.loc[shuffled_idxs[idxs_val1],fold_split_key] = 'val_1'
    full_data.loc[shuffled_idxs[idxs_val2],fold_split_key] = 'val_2'
    
    return full_data

# split the data into default 40% train 1, 40% train 2, 10% val 1, and 10% val 2
def split_and_save(csv_file, split_seeds, splits = [.4, .4, .1, .1], overwrite=False):      
    full_data = pd.read_csv(csv_file)
    csv_name = os.path.basename(csv_file)
        

    for seed in split_seeds:
        full_data = add_split_row(full_data, seed, splits=splits)
        
    fp_out = csv_file.replace('.csv', '_with_splits.csv')
    if (not overwrite) and os.path.exists(fp_out):
        print('wont overwrite {0} without while overwrite == False'.format(fp_out))
    else:
        print('writing to {0}'.format(fp_out))
        full_data.to_csv(fp_out)
                                          
    return fp_out        

def subsample_df_by_groups(full_data, 
                           group_key,
                           subset_groups, 
                           group_sizes, 
                           rs,
                           fold_key = 'fold',
                           keep_test_val=True, 
                           shuffle=False):
    
    if keep_test_val:
        data_to_split = full_data[full_data[fold_key] == 'train'].copy()
    
        # don't change the representation of val/test data
        val_test_data = full_data[full_data[fold_key] != 'train'].copy()
    
    else:
        data_to_split = full_data.copy()
        
    dfs_sampled_by_group = []                  
        
    for g, groups_this_subset in enumerate(subset_groups):
        # subset of the full dataset that belongs to this subset
        mask_this_subset = [False for x in range(len(data_to_split))]  
        for group_name in groups_this_subset:    
            mask_this_subset = np.logical_or(mask_this_subset,
                                             data_to_split[group_key].values == group_name)
            
        df_this_subset = data_to_split[mask_this_subset]
        
        # sample from this subset according to group_sizes_this_trial
        size_this_group = int(group_sizes[g])
                 
        sample_idxs = rs.choice(len(df_this_subset), 
                                size_this_group, 
                                replace=False)    
            
        dfs_sampled_by_group.append(df_this_subset.iloc[sample_idxs])
    
    if keep_test_val:
        dfs_sampled_by_group.append(val_test_data)
        
    if shuffle:
        return pd.concat(dfs_sampled_by_group, ignore_index=True).sample(frac=1)
    else:    
        return pd.concat(dfs_sampled_by_group, ignore_index=True)


def get_single_split(base_csv_file,
                     split_seed,
                     split=1,
                     save_fn = None,
                     write_split_file_first = False):
         
    
    split_csv_fp = base_csv_file.replace('.csv', '_with_splits.csv')
    
    full_data = pd.read_csv(split_csv_fp).copy()
    
    fold_key = 'fold_split_{}'.format(split_seed)
    
    # get train1 and val1 instances
    this_idxs_mask = [x in ['train_{}'.format(split), 'val_{}'.format(split)] for x in full_data[fold_key]]
    data_1 = full_data.loc[this_idxs_mask].copy()
    # rename to train and val
    data_1.loc[:,'fold'] = data_1.loc[:,fold_key].replace({'train_{}'.format(split):'train', 
                                  'val_{}'.format(split):'val'})
    
    if save_fn is not None:
        data_1.to_csv(save_fn)
        
    return data_1
    
def get_first_and_append_second_split(base_csv_file,
                                      group_key,
                                      seed,
                                      group_sizes_dict,
                                      write_split_file_first = False):
    
    rs = np.random.RandomState(seed)
    # group fracs should be a dict
    
    data_1 = get_single_split(base_csv_file, seed, split=1)
    data_2 = get_single_split(base_csv_file, seed, split=2)
    
    # subset the dataset 2
    subset_groups = [[int(x)] for x in group_sizes_dict.keys()]
    subset_sizes = np.array([[x] for x in group_sizes_dict.values()])
    data_2_subset = subsample_df_by_groups(data_2, 
                                           group_key,
                                           subset_groups, 
                                           subset_sizes,
                                           rs)
    
    data_2_subset.replace({'val': 'test'}, inplace=True)
    
    return pd.concat([data_1, data_2_subset], ignore_index=True)