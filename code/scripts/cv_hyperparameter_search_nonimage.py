import pandas as pd
import numpy as np
import sys
import os
import pickle
import argparse
import itertools
import scipy.sparse

import sklearn
import sklearn.metrics

# local imports
import train_fxns_nonimage
from train_fxns_nonimage import cv_subset_and_train
from dataset_params import dataset_params
data_dir = '../../data'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, 
                        help='string identifier of the dataset')
    parser.add_argument('--num_splits', type=int, default=5, 
                        help='number of seeds to run')
    parser.add_argument('--seed_beginning', type=int, default=0, 
                        help='seed to start with')
    parser.add_argument('--reweight', type=bool, default=False, 
                        help='whether to reweight')
    parser.add_argument('--pred_fxn_name', type=str, default=0, 
                        help='which pred function to run the cv for')
    parser.add_argument('--results_tag', type=str, default='debug', 
                        help='tag for results descriptor')
    
    
    args = parser.parse_args()
    num_cv_splits, seed_beginning = args.num_splits, args.seed_beginning
    dataset_name = args.dataset_name
    pred_fxn_name = args.pred_fxn_name
    reweight = args.reweight
    print("reweighting?",reweight)
    results_tag = args.results_tag
    
    results_general_path = '../../results/subset_results/'
    
    # do one random seed per split since there's num_splits splits
    num_seeds_each_split = 1
    
    fracs_group_a = np.array([0.02, 0.05, 0.2, 0.5, 0.8, 0.95, 0.98])
    
    if reweight:
        obj_str ='IW'
    else:
        obj_str = 'ERM'
        reweight_target_dist = None
    
    
    # the classifiers and the parameters searched will depend on the dataset
    # default to predicting 0/1 for binary targets unless told otherwise
    # should also be false when prediction targets outside [0,1]
    predict_probalities = False
    
    # get parameters for each dataset
    if dataset_name.lower() == 'goodreads':
        label_colname = 'rating'
        group_key = 'genre'
        all_group_colnames = ['history', 'fantasy']
        
        data_dir_goodreads = os.path.join(data_dir, 'goodreads')
        data_fn = os.path.join(data_dir_goodreads,
                               'goodreads_{0}_{1}_5_{2}_fold_splits.csv'.format(all_group_colnames[0],
                                                                           all_group_colnames[1],
                                                                              group_key))
        
        features_fn = data_fn.replace('5_{}_fold_splits.csv'.format(group_key), 'features_2k.npz')
        
        results_descriptor ='goodreads_2k_hpSel_'+obj_str
        
        results_descriptor = 'goodreads_2k_{0}_{1}_hpSel_'.format(all_group_colnames[0],
                                                                  all_group_colnames[1])+obj_str
        
        group_keys_to_stratify_cv = [group_key]
        
        data = pd.read_csv(data_fn)
        X = scipy.sparse.load_npz(features_fn).toarray()

        # set up the acc_keys
        acc_fxns = {'mse': sklearn.metrics.mean_squared_error,
                    'mae': sklearn.metrics.mean_absolute_error}
        
        # make a list of pred fxn names, or assume we're using one rom the command line argument
        if pred_fxn_name == 0:
            pred_fxn_names = ['ridge','logistic_regression', 'rf_regressor']
        else:
            pred_fxn_names = [pred_fxn_name]
    
        pred_fxn_dict = {'logistic_regression': train_fxns_nonimage.fit_logistic_regression_multiclass,
                         'rf_classifier': train_fxns_nonimage.fit_rf_classifier,
                         'rf_regressor': train_fxns_nonimage.fit_rf_regressor,
                         'ridge': train_fxns_nonimage.fit_ridge_regression}
        
        if reweight:
            pred_fxn_hps_to_search = {'logistic_regression': {'penalty': ['l2'],
                                                              'C': [10.0,1.0,0.1,0.01],
                                                             'solver': ['lbfgs']},
 #                                                            'C':[10.0]},
                                      'rf_classifier': {'max_depth': [32,64, 128], 
                                                        'n_estimators': [100,200,400]},
                                      'rf_regressor': {'max_depth': [32,64, 128], 
                                                       'n_estimators': [100,200,400],
                                                       'criterion': ['mse']},
                                      'ridge': {'alpha': [0.01,0.1,1.0,10.0,100.0]}
                                     }
        else:
            pred_fxn_hps_to_search = {'logistic_regression': {'penalty': ['l2'],#['l1','l2'],
                                                              'C': [0.01, 0.1,1.0, 10.0],
                                                             'solver': ['lbfgs']},
                                      'rf_classifier': {'max_depth': [32, 64, 128], 
                                                        'n_estimators': [100,200,400]},
                                      'rf_regressor': {'max_depth': [32,64, 128], 
                                                       'n_estimators': [100,200,400],
                                                       'criterion': ['mse']},
                                      'ridge': {'alpha': [0.01,0.1,1.0,10.0,100.0]}
                                     }
                                 
        if reweight:
            gamma = dataset_params[dataset_name.lower()]['gamma']
            reweight_target_dist = [gamma,1-gamma]
            print('reweighting to ',reweight_target_dist)
            
    elif dataset_name.lower() == 'adult':
        label_colname = '>50k'
        group_key = 'male'
        all_group_colnames = [0, 1]
        
        data_dir_adult = os.path.join(data_dir, 'adult')
        data_fn = os.path.join(data_dir_adult,
                               'df_adult_labels_5_male_fold_splits.csv')
        
        if results_tag == 'no_gender':
            features_fn = data_fn.replace('labels_5_male_fold_splits.csv', 'features_no_gender.csv')
            print('reading features from ', features_fn)
        else:
            features_fn = data_fn.replace('labels_5_male_fold_splits.csv', 'features.csv')
        
        
        results_descriptor = 'adult_subsetting_hpSel_{0}_{1}'.format(results_tag, obj_str)
        
        group_keys_to_stratify_cv = [group_key]
        
        data = pd.read_csv(data_fn)
        X = pd.read_csv(features_fn).to_numpy()
        
        print(X.shape)
        
        # set up the acc_keys
        acc_fxns = {'acc': sklearn.metrics.accuracy_score,
                    'auc_roc': sklearn.metrics.roc_auc_score}

        # loop through all pred fxns considered if non is specified
        if pred_fxn_name == 0:
            pred_fxn_names = ['ridge','logistic_regression', 'rf_classifier']
        else:
            pred_fxn_names = [pred_fxn_name]
    
        pred_fxn_dict = {'logistic_regression': train_fxns_nonimage.fit_logistic_regression,
                         'rf_classifier': train_fxns_nonimage.fit_rf_classifier,
                         'rf_regressor': train_fxns_nonimage.fit_rf_regressor,
                         'ridge': train_fxns_nonimage.fit_ridge_regression}

        pred_fxn_hps_to_search = {'logistic_regression': {'penalty': ['l2'],
                                                              'C': [0.001, 0.01, 0.1,1.0, 10.0],
                                                             'solver': ['lbfgs']},
                                      'rf_classifier': {'max_depth': [8,16,32], 
                                                        'n_estimators': [100, 200, 400]},
                                      'ridge': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}
                                     }
                                 
        if reweight:
            gamma = dataset_params[dataset_name.lower()]['gamma']
            reweight_target_dist = [gamma,1-gamma]
            print('reweighting to ',reweight_target_dist)
        
    elif dataset_name.lower() == 'mooc':
        label_colname = 'certified'
        group_key = 'post_secondary'
        all_group_colnames = [0, 1]
        
        data_dir_mooc = os.path.join(data_dir, 'mooc')
        data_fn = os.path.join(data_dir_mooc,
                               'df_mooc_labels_5_post_secondary_fold_splits.csv')
        
        features_fn = data_fn.replace('labels_5_post_secondary_fold_splits.csv', 'features_no_demographics.csv')
        results_tag = 'no_demographics'
        #features_fn = data_fn.replace('labels_5_post_secondary_fold_splits.csv', 'features_with_demographics.csv')
        #results_tag = 'with_demographics'
        
        
        results_descriptor = 'mooc_subsetting_post_secondary_hpSel_{0}_{1}'.format(results_tag, obj_str)
        
        group_keys_to_stratify_cv = [group_key]
        
        data = pd.read_csv(data_fn)
        X = pd.read_csv(features_fn).to_numpy()
        
        # set up the acc_keys
        acc_fxns = {'acc': sklearn.metrics.accuracy_score,
                    'auc_roc': sklearn.metrics.roc_auc_score}
        
        # make a list of pred fxn names, or assume we're using one rom the command line argument
        pred_fxn_names = ['ridge', 'rf_classifier','logistic_regression']
        if pred_fxn_name == 0:
            pred_fxn_names = ['ridge','logistic_regression', 'rf_classifier']
        else:
            pred_fxn_names = [pred_fxn_name]
    
        pred_fxn_dict = {'logistic_regression': train_fxns_nonimage.fit_logistic_regression,
                         'rf_classifier': train_fxns_nonimage.fit_rf_classifier,
                         'ridge': train_fxns_nonimage.fit_ridge_regression}
        
        
        pred_fxn_hps_to_search = {'ridge': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},
                                  'logistic_regression': {'penalty': ['l2'],
                                                          'C': [0.001, 0.01, 0.1,1.0, 10.0],
                                                          'solver': ['lbfgs']},
                                  'rf_classifier': {'max_depth': [4,8,16,32], 
                                                    'n_estimators': [100, 200, 400]}}
        
        predict_probalities = True
        pred_probs_by_pred_name = {'logistic_regression': True,
                                   'rf_classifier': True, 'ridge':False}
                                 
        if reweight:
            gamma = dataset_params[dataset_name.lower()]['gamma']
            reweight_target_dist = [gamma,1-gamma]
            print('reweighting to ',reweight_target_dist)
    else:
        print('TODO: need to input the data files for {0}'.format(dataset_name))
    
    # from fracs_group_a to subset sizes
    # pick the minimum group size in any training set across the 5 folds
    num_cv_splits = 5
    fold_keys = ['cv_fold_{0}'.format(i) for i in range(num_cv_splits)]
    n_train_per_group_all = [data[data[fold_key] == 'train'].groupby(group_key).count().iloc[:,0].values \
                         for fold_key in fold_keys]
    n_train_per_group = np.min(np.array(n_train_per_group_all))
    subset_sizes_0 = (n_train_per_group * fracs_group_a).astype(int)
    subset_sizes_cv = np.vstack([subset_sizes_0,
                                 n_train_per_group - subset_sizes_0])
    
    # instantiate fps where output data is to be stored
    this_results_path = os.path.join(results_general_path, results_descriptor)
    
    for p in [results_general_path,this_results_path]:
        if not os.path.exists(p):
            print(p,' did not exist; making it now')
            os.makedirs(p)
        
    
    # loop over the types of classifiers we want to consider
    for pred_fxn_name in pred_fxn_names:
        # get hps and such for this prediction function
        pred_fxn = pred_fxn_dict[pred_fxn_name]
        param_dict = pred_fxn_hps_to_search[pred_fxn_name]
        hp_param_keys = list(param_dict.keys())
        
        # if output dir doesn't exist for this pred function, create it
        pred_fxn_base_name = 'subset_{0}_5foldcv'.format(group_key)
        results_path_this_pred_fxn = os.path.join(this_results_path,pred_fxn_base_name, pred_fxn_name)
        if not os.path.exists(results_path_this_pred_fxn):
            print(results_path_this_pred_fxn,' did not exist; making it now')
            os.makedirs(results_path_this_pred_fxn)
        
        # loop over all hp configs for this classifier type
        for hp_setting in itertools.product(*list(param_dict.values())):
            # instantiate the model kwargs for this setting
            model_kwargs = {}
            for i, hp in enumerate(hp_setting):
                model_kwargs[hp_param_keys[i]] = hp
            print(pred_fxn_name, model_kwargs)
    
            # instantiate the string that will identify the results
            # compact way to write <param_key1>_<paramkey>_<param_key2>_<param2>_...
            kwargs_specifier = '_'.join(['_'.join([str(y) for y in x]) for x in model_kwargs.items()])
    
            if predict_probalities:
                pred_probs = pred_probs_by_pred_name[pred_fxn_name]
            else:
                pred_probs = False
            print('predicting probabilities: ',pred_probs)
            
            results = cv_subset_and_train(data, 
                                          X, 
                                          group_key, 
                                          label_colname, 
                                          subset_sizes_cv, 
                                          pred_fxn = pred_fxn,
                                          model_kwargs = model_kwargs,
                                          acc_fxns = acc_fxns,
                                          reweight = reweight,
                                          reweight_target_dist = reweight_target_dist,
                                          num_seeds = num_cv_splits,
                                          predict_prob = pred_probs,
                                          verbose=False)
                        
            accs_by_group, accs_total = results 
            
            # aggregate results for saving
            results_dict = {'group_key': group_key,
                            'num_cv_folds': 5,
                            'subset_sizes': subset_sizes_cv,
                            'accs_by_group': accs_by_group, 
                            'accs_total': accs_total}

            # save the 5fold results for this hyperparameter setting
            fn_save = os.path.join(results_path_this_pred_fxn,kwargs_specifier+'.pkl')
            print('saving results in ', fn_save)

            with open(fn_save, 'wb') as f:
                pickle.dump(results_dict, f)
