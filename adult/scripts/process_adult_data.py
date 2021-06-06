import pandas as pd
import numpy as np
import sklearn
import os
import sys

sys.path.append('../../code/scripts')
from dataset_chunking_fxns import add_stratified_kfold_splits

# Load data into pd dataframes and adjust feature names 
data_dir = '../../data/adult'
file_train = os.path.join(data_dir, 'adult.data')
file_test = os.path.join(data_dir, 'adult.test')

train_df = pd.read_csv(file_train, header=None, na_values='?')
test_df = pd.read_csv(file_test, header=None, na_values='?',skiprows=[0])

features = ['age', 'workclass', 'final-weight', 'education', 'education-num', 'marital-status', 'occupation', 
          'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']

      
train_df.columns = features
test_df.columns = features
print("Original number of points in training:", len(train_df))
print("Original number of points in test:", len(test_df))
print()

#drop final-weight feature because it's a measure of population proportion represented by the profile
train_df = train_df.drop(['final-weight'], axis=1)
test_df = test_df.drop(['final-weight'], axis=1)

feat_list = list(train_df.keys())
feat_list.remove('label')
print('number of features before one-hot encoding:', len(feat_list))

# train data: one hot encode non-binary discontinuous features
print("One hot encoding the following non-binary, discontinuous features:")
one_hot_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
for col in one_hot_columns:
    print(col)
print()
one_hot_workclass = pd.get_dummies(train_df['workclass'])
for feature in one_hot_columns:
    one_hot_encoding = pd.get_dummies(train_df[feature])
    if ' ?' in one_hot_encoding.columns:
        one_hot_encoding = one_hot_encoding.drop([' ?'], axis=1)
    train_df = train_df.join(one_hot_encoding)
train_df = train_df.drop(one_hot_columns, axis=1)


# train data: change binary features to 0/1  
binary_columns = ['sex', 'label']
for feature in binary_columns:
    one_hot_encoding = pd.get_dummies(train_df[feature])
    binary_encoding = one_hot_encoding.drop([one_hot_encoding.columns[0]], axis=1)
    train_df = train_df.join(binary_encoding)
train_df = train_df.drop(binary_columns, axis=1)
print('New name of train labels column:', train_df.columns[len(train_df.columns)-1])


# test data: one hot encode non-binary discontinuous features
one_hot_workclass = pd.get_dummies(test_df['workclass'])
for feature in one_hot_columns:
    one_hot_encoding = pd.get_dummies(test_df[feature])
    if ' ?' in one_hot_encoding.columns:
        one_hot_encoding = one_hot_encoding.drop([' ?'], axis=1)
    test_df = test_df.join(one_hot_encoding)
test_df = test_df.drop(one_hot_columns, axis=1)


# test data: change binary features to 0/1  
for feature in binary_columns:
    one_hot_encoding = pd.get_dummies(test_df[feature])
    binary_encoding = one_hot_encoding.drop([one_hot_encoding.columns[0]], axis=1)
    test_df = test_df.join(binary_encoding)
test_df = test_df.drop(binary_columns, axis=1)

# check for mismatches in hot encoded features between train and test data
def check_features(train_df, test_df):
    for i in range(len(train_df.columns)):
        if train_df.columns[i] != test_df.columns[i]:
            print('Mismatch on', train_df.columns[i], ' from train data and', test_df.columns[i], 
                  ' from test data.')
            print('The next train feature is', train_df.columns[i+1], ' and the next test feature is', 
                  test_df.columns[i+1], '.')
            break
        if i == len(train_df.columns) - 1: 
            print('All test and train features match')
    print()
# print('At least', str(len(train_df.columns) - len(test_df.columns)), 'difference(s) in features labels.')
# check_features(train_df, test_df)

# manually add the missing column(s)
test_df[' Holand-Netherlands'] = 0
test_df = test_df.rename({' >50K.': ' >50K'}, axis=1)
test_df = test_df[train_df.columns]
print('New name of test labels column:', test_df.columns[len(test_df.columns)-1])
print()
check_features(train_df, test_df)

# remove NaNs
print("Dropping rows with NaNs")
test_df.dropna()
train_df.dropna()
print("New number of points in training:", len(train_df))
print("New number of points in test:", len(test_df))
combined = len(train_df) + len(test_df)
print()

# shuffle the training data
print("Shuffling training data")
print()
np.random.seed(11)
train_df = train_df.sample(frac=1).reset_index(drop=True)

formatted_columns = [s.strip().lower() for s in train_df.columns]
train_df.columns = formatted_columns
test_df.columns = formatted_columns

print('\033[1m' + 'Label:' + '\033[0m')
print(train_df.columns[-1])
print()
print('\033[1m' + 'Features:' + '\033[0m')

for f in train_df.columns[:-1]:
    print(f)
print()

# now, save the data as one CSV
n = train_df.shape[0]
m = test_df.shape[0]

train_list = []
for i in range(n):
    train_list += ['train']
    
test_list = []
for i in range(m):
    test_list += ['test']
    
train_df['fold'] = train_list
test_df['fold'] = test_list


data = pd.concat([train_df, test_df])

project_data_dir = '../../data'

out_dir_data = os.path.join(project_data_dir,'adult')

#if out dir doesn't exist, make it
if not os.path.exists(out_dir_data):
        os.mkdir(out_dir_data)

# add indices to features
data['X_idxs'] = np.arange(len(data))

out_csv_fp = os.path.join(out_dir_data, 'df_adult_labels.csv')
if not os.path.exists(out_csv_fp):
    data.to_csv(out_csv_fp, index=False)
    print("Saved data as df_adult_labels.csv")
else:
    print(out_csv_fp, ' exists, wont overwrite')

# create separate features file
features = data.drop(['X_idxs', 'fold', '>50k'], axis=1)
out_csv_fp = os.path.join(out_dir_data, 'df_adult_features.csv')

print('number of features after one-hot encoding:', features.shape[1])


if not os.path.exists(out_csv_fp):
    features.to_csv(out_csv_fp, index=False)
    print("Saved features as", out_csv_fp)
else:
    print(out_csv_fp, ' exists, wont overwrite')
    
out_csv_fp_no_gender = os.path.join(out_dir_data, 'df_adult_features_no_gender.csv')
#TODO: drop all keys that imply gender, not just male
gender_related_keys = ['male', 'husband', 'wife']
features_no_gender = features.drop(gender_related_keys, axis=1)
if not os.path.exists(out_csv_fp_no_gender):
    features_no_gender.to_csv(out_csv_fp_no_gender, index=False)
    print("Saved features as ", out_csv_fp_no_gender)
else:
    print(out_csv_fp_no_gender, ' exists, wont overwrite')
    
    

print("total gamma for female:", 1-data['male'].mean())
print("train gamma for female:", 1-train_df['male'].mean())
print("test gamma for female:", 1-test_df['male'].mean())

# add stratified kfold splits and save
data_both_with_cv_splits = add_stratified_kfold_splits(os.path.join(out_dir_data, 'df_adult_labels.csv'),
                                                       'male',
                                                        num_splits=5,
                                                        overwrite=False)

