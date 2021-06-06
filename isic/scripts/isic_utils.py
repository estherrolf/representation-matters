import pandas as pd
import pickle
import json

def isic_study_id_to_name(data_file='../data/isic/image_labels_no_nans_10_plus.csv'):
    original_data = pd.read_csv(data_file)

    study_id_to_name = {}
    for study_id in original_data['study_name_id'].unique():
        study_id_to_name[study_id] = original_data[original_data['study_name_id'] == study_id]['study_name'].unique()[0]
    
    return study_id_to_name

def isic_study_name_to_id(metadata_file='../data/isic/image_labels_no_nans_10_plus_metadata.pkl'):
    with open(metadata_file, 'rb') as f:
        metadata_files =  json.load(f)
    
    return metadata_files['study_name_to_id']

def isic_age_id_to_age(metadata_file='../data/isic/image_labels_no_nans_10_plus_metadata.pkl'):
    with open(metadata_file, 'rb') as f:
        metadata_files =  json.load(f)
        
    # reverse the dict.
    age_id_to_age_dict = {}
    for key, value in metadata_files['age_to_id'].items():
        age_id_to_age_dict[value] = key
        
    return age_id_to_age_dict


def read_results_by_seed(file_format, seeds, group_key):
    
    results_by_group_by_trial_basic = pd.DataFrame()
    for t in seeds:    
        with open(file_format.format(t), 'rb') as f:
            data_this = pickle.load(f)
            
            accs_by_group, accs_all = data_this
            
            accs_by_group['trial'] = t
        
            results_by_group_by_trial_basic = pd.concat((results_by_group_by_trial_basic, 
                                               accs_by_group.reset_index(level=0)))
            
    results_by_group = results_by_group_by_trial_basic.groupby(group_key)
    
    return results_by_group
    
    
    