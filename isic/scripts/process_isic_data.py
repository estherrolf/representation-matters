import json
import os

import pandas as pd
import numpy as np
import isic_utils
import compile_isic_data

import sys
sys.path.append('../../code/scripts')
import dataset_chunking_fxns
import make_smaller_image_folder

data_dir = isic_data_dir = '../../data' 
isic_data_dir = os.path.join(data_dir,'isic')
out_dir_data = isic_data_dir
out_dir_metadata = os.path.join(isic_data_dir,'int_metadata')
    
# add ids
def add_numeric_group_ids(df, group_keys):
    metadata = {}

    for key in group_keys:
        key_id = key + '_id'

        # add an age id column
        key_to_keyid = {}
        key_to_keyid_safe = {}
        
        for i,group in enumerate(np.sort(df[key].unique())):
            key_to_keyid[group] = i
            key_to_keyid_safe[str(group)] = i

        df[key_id] = df[key].apply(lambda row: key_to_keyid[row])
        
        metadata[key+"_to_id"] = key_to_keyid_safe
        
    return df, metadata

def floor_to_tens(x):
        return 10 * np.floor(x/10)
    
def process_isic(isic_data_dir, do_sonic=False):
    
    isic_csv_dir = os.path.join(isic_data_dir,'isic_unprocessed.csv')
    
    df_unprocessed = pd.read_csv(isic_csv_dir)
    df = df_unprocessed.copy()
    
    # add binary targets
    df['benign_malignant'].replace({'indeterminate': np.nan,
                                    'indeterminate/benign': np.nan,
                                    'indeterminate/malignant': np.nan
                                   }, inplace=True)

    df.dropna(subset=['benign_malignant', 'age_approx', 'sex'], how='any',inplace=True)

    df['benign_malignant_01'] = df['benign_malignant'].replace({'benign':0, 'malignant':1,})

    # add age approx by decade
    df.loc[:,'age_approx_by_decade'] = df.loc[:,'age_approx'].apply(lambda row: floor_to_tens(row))
    df.loc[:,'age_over_45'] = (df.loc[:,'age_approx'] > 45).astype(int)
    df.loc[:,'age_over_50'] = (df.loc[:,'age_approx'] > 50).astype(int)
    df.loc[:,'age_over_55'] = (df.loc[:,'age_approx'] > 55).astype(int)
    df.loc[:,'age_over_60'] = (df.loc[:,'age_approx'] > 60).astype(int)
    
    # aggregate study names, though note that the substudies have unique characteristics within MSK/UDA
    df.loc[:,'study_name_aggregated'] = df.loc[:,'study_name'].replace({'MSK-1': 'MSK',
                                                                        'MSK-2': 'MSK',
                                                                        'MSK-3': 'MSK',
                                                                        'MSK-4': 'MSK',
                                                                        'MSK-5': 'MSK',
                                                                        'UDA-1': 'UDA',
                                                                        'UDA-2': 'UDA'})

    # clean anatomy site
    df['anatom_site_general'].fillna('unknown', inplace=True)
    
    # keep track of with and without isic separatey
    df_no_sonic = df.copy()[df['study_name'] != 'SONIC']
    df_just_sonic = df.copy()[df['study_name'] == 'SONIC']

    # add folds separately for sonic - outdated, we now do this later
#     df_no_sonic = add_train_val_test_splits(df_no_sonic)
#     df_just_sonic = add_train_val_test_splits(df_just_sonic)
    df_with_sonic = pd.concat([df_no_sonic,df_just_sonic])
    
    # keys that need numeric ids, that we might want to stratify evaluation on. 
    possible_group_keys = ['study_name', 
                           'study_name_aggregated',
                           'age_approx_by_decade',
                           'age_over_45', 
                           'age_over_50', 
                           'age_over_55', 
                           'age_over_60', 
                           'anatom_site_general', 
                           'sex']

    df_no_sonic, metadata_no_sonic = add_numeric_group_ids(df_no_sonic, possible_group_keys)
    df_with_sonic, metadata_with_sonic = add_numeric_group_ids(df_with_sonic, possible_group_keys)


    # save the data - for now only save data with no sonic
    df_no_sonic.reset_index(inplace=True, drop=True)
    no_sonic_csv_fn = os.path.join(out_dir_data, 'df_no_sonic.csv')
    df_no_sonic.to_csv(no_sonic_csv_fn,index=False)

    with open(os.path.join(out_dir_metadata, 'df_no_sonic.json'), 'w') as f:
        json.dump(metadata_no_sonic, f)
        
    # add test and train splits. 
    for group_key in ['age_over_50_id']:
        df_this_key = df_no_sonic.copy()
        df_with_test = dataset_chunking_fxns.add_stratified_test_splits(df_this_key, group_key)
        new_csv_name = no_sonic_csv_fn.replace('.csv', '_{0}.csv'.format(group_key))
        df_with_test.to_csv(new_csv_name, index=False)

        dataset_chunking_fxns.add_stratified_kfold_splits(new_csv_name, group_key, overwrite=False)

    # Keep the following lines in case we want to look at adding sonic back in
    # add test and train splits. 
    if do_sonic:
        df_with_sonic.reset_index(inplace=True, drop=True)
        with_sonic_csv_fn = os.path.join(out_dir_data, 'df_with_sonic.csv')
        # study name aggregated id will be subsetted also by this
        for group_key in ['study_name_id']:
            df_this_key = df_with_sonic.copy()
          #  print(df_this_key.keys())
            df_with_test = dataset_chunking_fxns.add_stratified_test_splits(df_this_key, group_key)
            new_csv_name = with_sonic_csv_fn.replace('.csv', '_{0}.csv'.format(group_key))
            df_with_test.to_csv(new_csv_name, index=False)
            
            dataset_chunking_fxns.add_stratified_kfold_splits(new_csv_name, group_key, overwrite=False)

        with open(os.path.join(out_dir_metadata, 'df_with_sonic.json'), 'w') as f:
            json.dump(metadata_with_sonic, f)

if __name__ == "__main__":
    # make sure all the output paths exists
    for d in [out_dir_data, out_dir_metadata]:
        if not os.path.exists(d):
            os.mkdir(d)
            
    # if the compiled data file doesn't already exist, make it
    if not os.path.exists(os.path.join(isic_data_dir,'isic_unprocessed.csv')):
        print('first compiling dataset to a csv')
        compile_isic_data.compile_instances()
        
    # now process and store the data
    do_sonic = True
    process_isic(isic_data_dir, do_sonic=do_sonic)
    
    if do_sonic:
        print('making smaller image folder for isic with sonic')
        make_smaller_image_folder.main(csv_filename='isic/df_with_sonic_study_name_id.csv',
                                       include_sonic=True)
    
    # always do the other
    print('making smaller image folder for isic without sonic')
    make_smaller_image_folder.main(csv_filename='isic/df_no_sonic_age_over_50_id.csv',
                                   include_sonic=False)
