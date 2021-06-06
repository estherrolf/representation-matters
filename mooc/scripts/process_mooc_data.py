import pandas as pd
import numpy as np
import os
import sys

sys.path.append('../../code/scripts')
from dataset_chunking_fxns import add_stratified_kfold_splits, add_stratified_test_splits

# open the harvard/mit data
file_dir = '../../data'
file_dir = os.path.join(file_dir, 'mooc')
file_path = os.path.join(file_dir,'harvard_mit.csv')
all_df = pd.read_csv(file_path, header=None, na_values='NaN', low_memory=False)
features = ["course_id", "userid_DI", "registered", "viewed", "explored", "certified", "final_cc_cname_DI", 
            "LoE_DI", "YoB", "gender", "grade", "start_time_DI", "last_event_DI", "nevents", "ndays_act", 
            "nplay_video", "nchapters", "nforum_posts", "roles", "incomplete_flag"]
all_df.columns = features
all_df = all_df.iloc[1:]
all_df.reset_index(drop=True)

# only keep the following features
print("Dropping nans based on the following features: ")
features_drop_na = ['gender', 'LoE_DI', 'final_cc_cname_DI', 'YoB', 'ndays_act', 'nplay_video', 'nforum_posts', 
                        'nevents', 'course_id', 'certified']

for f in features_drop_na:
    print(f)
print()

subset_all_df = all_df[features_drop_na]
print('original number of points:',len(subset_all_df))
subset_all_df = subset_all_df.dropna()
subset_all_df = subset_all_df.reset_index(drop=True)
print()


print('One hot encoding course id, education level, and country...')
print()
# change gender labels from 'M' or 'F' to binary
gender_to_binary = {'m':0, 'f':1}
subset_all_df['gender'] = subset_all_df['gender'].apply(func=(lambda g: gender_to_binary[g]))

# one hot encode course id
one_hot_encoding_course_id = pd.get_dummies(subset_all_df['course_id'])
course_id_keys = list(one_hot_encoding_course_id.keys())
subset_all_df = subset_all_df.join(one_hot_encoding_course_id)

# # one hot encode education level
# one_hot_encoding_edu = pd.get_dummies(subset_all_df['LoE_DI'])
# subset_all_df = subset_all_df.join(one_hot_encoding_edu)
# subset_all_df = subset_all_df.drop(['LoE_DI'], axis=1)

# change education level to be binary on whether the individual has completed any post-secondary education
def edu_to_binary(label):
    if label == 'Master\'s':
        return 1
    elif label == 'Secondary':
        return 0
    elif label == 'Less than Secondary':
        return 0
    elif label == 'Bachelor\'s':
        return 1
    elif label == 'Doctorate':
        return 1
    else:
        print('Unexpected input:', label)

# one hot encode eduation label        
one_hot_encoding_edu = pd.get_dummies(subset_all_df['LoE_DI'])
edu_keys = list(one_hot_encoding_edu.keys())
subset_all_df = subset_all_df.join(one_hot_encoding_edu)

# binary edu label for groups
subset_all_df['LoE_DI'] = subset_all_df['LoE_DI'].apply(edu_to_binary)
subset_all_df = subset_all_df.rename({'LoE_DI': 'post_secondary'}, axis='columns')


# one hot encode country
one_hot_encoding_country = pd.get_dummies(subset_all_df['final_cc_cname_DI'])
country_keys = list(one_hot_encoding_country.keys())
subset_all_df = subset_all_df.join(one_hot_encoding_country)
subset_all_df = subset_all_df.drop(['final_cc_cname_DI'], axis=1)

# shuffle
print("Shuffling data...")
print()
subset_all_df = subset_all_df.sample(frac=1).reset_index(drop=True)


# convert all number values to types that are considered numeric
#subset_all_df['grade'] = pd.to_numeric(subset_all_df['grade'],errors='coerce')
# 
min_year = int(subset_all_df['YoB'].min())
yob_key_new = 'YoB (from {0})'.format(min_year)
subset_all_df[yob_key_new] = pd.to_numeric(subset_all_df['YoB'],errors='coerce') - min_year
subset_all_df = subset_all_df.drop(['YoB'], axis=1)
subset_all_df['ndays_act'] = pd.to_numeric(subset_all_df['ndays_act'],errors='coerce')
subset_all_df['nplay_video'] = pd.to_numeric(subset_all_df['nplay_video'],errors='coerce')
subset_all_df['nforum_posts'] = pd.to_numeric(subset_all_df['nforum_posts'],errors='coerce')
subset_all_df['nevents'] = pd.to_numeric(subset_all_df['nevents'],errors='coerce')
subset_all_df['certified'] = pd.to_numeric(subset_all_df['certified'],errors='coerce')
subset_all_df = subset_all_df.dropna()
subset_all_df = subset_all_df.reset_index(drop=True)



# # calculate the certification stats
# overall_avg_grade = subset_all_df['certified'].sum() / len(subset_all_df)
# print('Overall certification rate:', overall_avg_grade)

# train_avg_grade = subset_all_df.iloc[:19200]['certified'].sum() / 19200
# print('Train certification rate:', train_avg_grade)

# test_avg_grade = subset_all_df.iloc[19200:]['certified'].sum() / (len(subset_all_df) - 19200)
# print('Test certification rate:', test_avg_grade)
# print()

# print features
# cols = subset_all_df.columns
# print("Features:")
# for feature in cols:
#     print(feature)
# print()

# num_female = subset_all_df.iloc[:19200]['gender'].sum()
# num_male = len(subset_all_df.iloc[:19200]['gender']) - num_female
# gender_gamma = num_male / (num_male + num_female)
# print("overall gamma for gender:", gender_gamma)

# num_psec = subset_all_df.iloc[:19200]['post_secondary'].sum()
# num_npsec = len(subset_all_df.iloc[:19200]['post_secondary']) - num_psec
# sec_gamma = num_npsec / (num_psec + num_npsec)
# print("overall gamma for no post secondary education:", sec_gamma)

# now, add train/test folds and save to csv 
# n = len(subset_all_df) - len(subset_all_df)// 5 
# m = len(subset_all_df) // 5

# fold_list = []
# for i in range(n):
#     fold_list += ['train']
    
# for i in range(m):
#     fold_list += ['test']
    
# subset_all_df['fold'] = fold_list

subset_all_df['fold'] = 0
subset_all_df = add_stratified_test_splits(subset_all_df,'post_secondary')


#dataframe[dataframe['Percentage'] > 80]
# num_psec = subset_all_df[subset_all_df['fold']=='test'].iloc[:19200]['post_secondary'].sum()
# num_npsec = len(subset_all_df[subset_all_df['fold']=='test'].iloc[:19200]['post_secondary']) - num_psec
# sec_gamma = num_npsec / (num_psec + num_npsec)
# print("test gamma for no post secondary education:", sec_gamma)

project_data_dir = '../../data'

out_dir_data = os.path.join(project_data_dir,'mooc')

#if out dir doesn't exist, make it
if not os.path.exists(out_dir_data):
        os.mkdir(out_dir_data)
        
subset_all_df['X_idxs'] = np.arange(len(subset_all_df))
out_csv_fp = os.path.join(out_dir_data, 'df_mooc_labels.csv')
if not os.path.exists(out_csv_fp):
    subset_all_df.to_csv(out_csv_fp, index=False)
    print("Saved data as df_mooc_labels.csv")
else:
    print(out_csv_fp, ' exists, wont overwrite')

#features = subset_all_df.drop(['X_idxs', 'fold', 'certified'], axis=1)

feature_names_no_demographics = ['ndays_act', 'nplay_video', 'nforum_posts', 'nevents'] + course_id_keys
feature_names_demographics_only = ['gender', yob_key_new] + country_keys + edu_keys
feature_names_with_demographics = feature_names_no_demographics  + feature_names_demographics_only

features_with_demographics = subset_all_df[feature_names_with_demographics]

print('number of points and features after dropna (inc. demographics):', features_with_demographics.shape)
print()

out_csv_fp = os.path.join(out_dir_data, 'df_mooc_features_with_demographics.csv')
if not os.path.exists(out_csv_fp):
    features_with_demographics.to_csv(out_csv_fp, index=False)
    print("Saved features as ",out_csv_fp)
else:
    print(out_csv_fp, ' exists, wont overwrite')
    

features_without_demographics = subset_all_df[feature_names_no_demographics]

print('number of points and features after dropna (not inc. demographics):', features_without_demographics.shape)
print()

out_csv_fp = os.path.join(out_dir_data, 'df_mooc_features_no_demographics.csv')
if not os.path.exists(out_csv_fp):
    features_without_demographics.to_csv(out_csv_fp, index=False)
else:
    print(out_csv_fp, ' exists, wont overwrite')    

    
# add stratified kfold splits and save
data_both_with_cv_splits = add_stratified_kfold_splits(os.path.join(out_dir_data, 'df_mooc_labels.csv'),
                                                       'post_secondary',
                                                        num_splits=5,
                                                        overwrite=False)
print()
folds = ['cv_fold_0', 'cv_fold_1', 'cv_fold_2', 'cv_fold_3', 'cv_fold_4']
for f in folds:
    num_psec = data_both_with_cv_splits.loc[data_both_with_cv_splits[f]=='train']['post_secondary'].sum()
    sec_gamma = 1 - num_psec / len(data_both_with_cv_splits.loc[data_both_with_cv_splits[f]=='train']['post_secondary'])
    print(f+" train gamma for no post secondary education:", sec_gamma)

    num_psec = data_both_with_cv_splits.loc[data_both_with_cv_splits[f]=='val']['post_secondary'].sum()
    sec_gamma = 1 - num_psec / len(data_both_with_cv_splits.loc[data_both_with_cv_splits[f]=='val']['post_secondary'])
    print(f+" val gamma for no post secondary education:", sec_gamma)





