import pandas as pd
import numpy as np
import sys
import os
import pickle
import argparse
import itertools
import time
import scipy.sparse

import sklearn
import sklearn.metrics

# local imports
import train_fxns_nonimage
from train_fxns_nonimage import subset_and_train
from dataset_params import dataset_params

data_dir = '../../data'
    
def kwargs_to_kwargs_specifier(model_kwargs):
    return '_'.join(['_'.join([str(y) for y in x]) for x in  model_kwargs.items()])
    
def fracs_to_subset_sizes_additional(frac_one_group,subset_sizes_0):
    assert frac_one_group <= 1 and frac_one_group >= 0
    sizes_smaller = subset_sizes_0*frac_one_group
    return np.hstack((np.vstack([subset_sizes_0, sizes_smaller]),
                      np.vstack([sizes_smaller,subset_sizes_0])))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, 
                        help='string identifier of the dataset')
    parser.add_argument('--num_seeds', type=int, default=10, 
                        help='number of seeds to run')
    parser.add_argument('--seed_beginning', type=int, default=0, 
                        help='seed to start with')
    parser.add_argument('--run_type', type=str, default='subsetting', 
                        help='subsetting, additional, or additional_equal_group_sizes')
    parser.add_argument('--reweight', type=bool, default=False, 
                        help='whether to reweight data to the pop proportions')
    parser.add_argument('--pred_fxn_name', type=str, default='rf_classifier', 
                        help='which pred function to run the cv for')
    parser.add_argument('--results_tag', type=str, default='debug', 
                        help='tag for results descriptor')
    
    
    args = parser.parse_args()
    num_seeds, seed_beginning = args.num_seeds, args.seed_beginning
    dataset_name = args.dataset_name
    run_type = args.run_type
    print('run type',run_type)
    pred_fxn_name = args.pred_fxn_name
    print('pred_fxn_name is:', pred_fxn_name)
    results_tag = args.results_tag
    reweight = args.reweight
    print('reweight is:', reweight)
    
    results_general_path = '../../results/subset_results/'
    
    # do ten random seeds
    num_eval_seeds = num_seeds
    seed_start = seed_beginning
        
    
    if reweight:
        obj_str ='IW'
    else:
        obj_str = 'ERM'
        reweight_target_dist = None
    
    if run_type == 'subsetting':
        if reweight:
            fracs_group_a = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, .3, 0.35, .4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99])
            #fracs_group_a = np.array([0.02, 0.05, 0.2, 0.5, 0.8, 0.95, 0.98])
        else:
            fracs_group_a = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, .3, 0.35, .4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0])
                    
    elif run_type == 'additional':
        print('additional')
        # determines the relative group sizes for the additonal experiment
        fracs_group_a = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, .3, .4,  0.5,  0.6, 0.7, 0.8, 0.9, 1.0])
        fracs_smaller = [0.125,0.25,0.5]
    
        
    elif run_type == 'additional_equal_group_sizes':  
        print('additional_equal_group_sizes')
        fracs_group_a = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, .3, .4,  0.5,  0.6, 0.7, 0.8, 0.9, 1.0])
        fracs_smaller = [1.0]
          
#    elif run_type == 'additional_max_one_group':  
#         fracs_group_a = np.array([0.1, 0.2, 0.3, 0.4,  0.5,  0.6, 0.7, 0.8, 0.9, 1.0])
    else:
        print('run type {0} not understood'.format(run_type))
        
    data = None
    X = None
    label_colname = ''
    group_key = ''
    acc_fxns = {}
    pred_fxn = None
    model_kwargs = {}
    results_descriptor = ''
    reweight_target_dist = []
    
    # default to false
    predict_prob = False
    
    # get parameters for each dataset
    if dataset_name.lower() == 'goodreads':
        label_colname = 'rating'
        
        all_group_colnames = ['history', 'fantasy']
        group_key = 'genre'
        
        data_dir_goodreads = os.path.join(data_dir, 'goodreads')
        data_fn = os.path.join(data_dir_goodreads,
                               'goodreads_{0}_{1}_5_{2}_fold_splits.csv'.format(all_group_colnames[0],
                                                                           all_group_colnames[1],
                                                                               group_key))
        
        features_fn =  data_fn.replace('5_{0}_fold_splits.csv'.format(group_key), 'features_2k.npz')
        
        data = pd.read_csv(data_fn)
        X = scipy.sparse.load_npz(features_fn).toarray()
        
        # set up the acc_keys
        acc_fxns = {'mse': sklearn.metrics.mean_squared_error,
                    'mae': sklearn.metrics.mean_absolute_error}
          
        pred_fxn_name = 'logistic_regression'
        pred_fxn = train_fxns_nonimage.fit_logistic_regression_multiclass
        
        if pred_fxn_name == 'logistic_regression':
            model_kwargs = {'penalty': 'l2','C':1.0, 'solver':'lbfgs'}
        elif pred_fxn_name == 'ridge':
            if reweight:
                def model_kwargs_by_alphas(alphas_both):
                    alpha_thresh = 0.05
                    if np.min(alphas_both) <= alpha_thresh:
                        return {'alpha': 1.0}
                    else:
                        return {'alpha': 0.1}
            else:
                model_kwargs = {'alpha': 0.1}
            
               
        results_descriptor = 'goodreads_2k_{1}_{2}_{0}_'.format(run_type,
                                                                all_group_colnames[0],
                                                                all_group_colnames[1])+obj_str
        
        if reweight:
            gamma = dataset_params[dataset_name.lower()]['gamma']
            reweight_target_dist = [gamma,1-gamma]
            print('reweighting to ',reweight_target_dist)
            
        # don't use this since params can vary by allocation
        if reweight:
            kwargs_specifier = 'subset'
        else:
            kwargs_specifier = kwargs_to_kwargs_specifier(model_kwargs)
    
    elif dataset_name.lower() == 'adult':
        label_colname = '>50k'
        group_key = 'male'
        data_dir_goodreads = os.path.join(data_dir, 'adult')
        data_fn = os.path.join(data_dir_goodreads, 'df_adult_labels.csv')
            
        if results_tag == 'no_gender':
            features_fn = data_fn.replace('labels.csv', 'features_no_gender.csv')
            print('reading features from ', features_fn)
        else:
            features_fn = data_fn.replace('labels.csv', 'features.csv')
            
        
        data = pd.read_csv(data_fn)
        X = pd.read_csv(features_fn).to_numpy()

        # set up the acc_keys
        acc_fxns = {'acc': sklearn.metrics.accuracy_score,
                    'auc_roc': sklearn.metrics.roc_auc_score}
        
        pred_fxn_name = 'rf_classifier'
        pred_fxn = train_fxns_nonimage.fit_rf_classifier

        if results_tag == 'no_gender':
            model_kwargs = {'max_depth': 16, 'n_estimators': 400}
        else:
            model_kwargs = {'max_depth': 16, 'n_estimators': 200}
            
        results_descriptor = 'adult_{0}_{1}_{2}'.format(run_type, results_tag, obj_str)
        
        if reweight:
            gamma = dataset_params[dataset_name.lower()]['gamma']
            reweight_target_dist = [gamma,1-gamma]
            print('reweighting to ',reweight_target_dist)  
            
        kwargs_specifier = kwargs_to_kwargs_specifier(model_kwargs)
            
    elif dataset_name.lower() == 'mooc':
        label_colname = 'certified'
        group_key = 'post_secondary'
        all_group_colnames = [0, 1]
        
        data_dir_goodreads = os.path.join(data_dir, 'mooc')
        data_fn = os.path.join(data_dir_goodreads,
                               'df_mooc_labels_5_post_secondary_fold_splits.csv')        
        

        if results_tag == 'no_demographics':
            features_fn = data_fn.replace('labels_5_post_secondary_fold_splits.csv', \
                                               'features_{0}.csv'.format(results_tag))
        else:
            features_fn = data_fn.replace('labels_5_post_secondary_fold_splits.csv', \
                                               'features_{0}.csv'.format('with_demographics'))
        
        
        data = pd.read_csv(data_fn)
        X = pd.read_csv(features_fn).to_numpy()
                        
        
        # set up the acc_keys
        acc_fxns = {'acc': sklearn.metrics.accuracy_score,
                    'auc_roc': sklearn.metrics.roc_auc_score}
        
        pred_fxn_name = 'rf_classifier'
        pred_fxn = train_fxns_nonimage.fit_rf_classifier
        
        if results_tag == 'no_demographics':
            model_kwargs = {'max_depth': 8, 'n_estimators': 400}
        else:
            model_kwargs = {'max_depth': 16, 'n_estimators': 400}
        predict_prob = True
        
        results_descriptor = '{3}_{0}_{1}_{2}'.format(run_type, results_tag, obj_str, dataset_name.lower())
        
        
        if reweight:
            gamma = dataset_params[dataset_name.lower()]['gamma']
            reweight_target_dist = [gamma,1-gamma]
            print('reweighting to ',reweight_target_dist)  
        
        kwargs_specifier = kwargs_to_kwargs_specifier(model_kwargs)
    else:
        print('TODO: need to input the data files for {0}'.format(dataset_name))
    
    # from fracs_group_a to subset sizes
    # pick the minimum group size in the traiing set
    fold_key = 'fold'
    n_train_per_group_all = data[data[fold_key] == 'train'].groupby(group_key).count().iloc[:,0].values
    n_train_per_group = np.min(np.array(n_train_per_group_all))
    print('Number of total points per group:', n_train_per_group)
    subset_sizes_0 = (n_train_per_group * fracs_group_a).astype(int)
    
    if run_type == 'subsetting':
        subset_sizes_subsetting = np.vstack([subset_sizes_0, n_train_per_group - subset_sizes_0])
        
        if reweight and pred_fxn_name == 'ridge':
            model_kwargs = [model_kwargs_by_alphas([a, 1-a]) for a in fracs_group_a]
        # otherwise they're already instantiated
    
    elif run_type == 'additional' or run_type == 'additional_equal_group_sizes' or run_type == 'additional_2':
        subset_sizes_additional = []
        for frac_smaller in fracs_smaller:
            ss = fracs_to_subset_sizes_additional(frac_smaller,subset_sizes_0)
            subset_sizes_additional.append(ss)
            
        subset_sizes_subsetting = np.hstack(subset_sizes_additional).astype(int)
        
        # don't need to do the maxed one twice
        subset_sizes_subsetting = np.unique(subset_sizes_subsetting, axis=1)
        print(subset_sizes_subsetting.shape)
        
    elif run_type == 'additional_max_one_group': 
        all_maxed = n_train_per_group * np.ones(len(subset_sizes_0))
        subset_sizes_subsetting = np.hstack((np.vstack([subset_sizes_0, all_maxed]),
                                             np.vstack([all_maxed, subset_sizes_0]))).astype(int)
        subset_sizes_subsetting = np.unique(subset_sizes_subsetting, axis=1)
        print(subset_sizes_subsetting.shape)
        
    else:                                                   
        print('run_type {0} not recognized'.format(run_type))
        quit()
              
    # instantiate fps where output data is to be stored
    this_results_path = os.path.join(results_general_path, results_descriptor)
    
    # if output dir doesn't exist for this pred function, create it
    pred_fxn_base_name = 'subset_{0}'.format(group_key)
    this_results_path_pre = os.path.join(this_results_path,pred_fxn_base_name)
    results_path_this_pred_fxn = os.path.join(this_results_path_pre, pred_fxn_name)
    
    for p in [results_general_path,this_results_path,this_results_path_pre,results_path_this_pred_fxn]:
        if not os.path.exists(p):
            print(p,' did not exist; making it now')
            os.makedirs(p)
        
    
                
    for s in range(seed_start,seed_start+num_eval_seeds):
        t1 = time.time()
        print('seed: ',s)
        results = subset_and_train(data, 
                                  X, 
                                  group_key, 
                                  label_colname, 
                                  subset_sizes_subsetting, 
                                  pred_fxn = pred_fxn,
                                  model_kwargs = model_kwargs,
                                  acc_fxns = acc_fxns,
                                  reweight = reweight,
                                  reweight_target_dist = reweight_target_dist,
                                  fold_key = 'fold',
                                  eval_key = 'test',
                                  seed_start = s,
                                  predict_prob = predict_prob,
                                  num_seeds = 1,
                                  verbose=False)
                        
        accs_by_group, accs_total = results 

        # aggregate results for saving
        results_dict = {'group_key': group_key,
                        'seed_start': seed_start,
                        'num_seeds': num_eval_seeds,
                        'subset_sizes': subset_sizes_subsetting,
                        'accs_by_group': accs_by_group, 
                        'accs_total': accs_total}

        # save the eval results for this hyperparameter setting
        fn_save = os.path.join(results_path_this_pred_fxn,kwargs_specifier+'_seed_{0}.pkl'.format(s))
        print('saving overall results results in ', fn_save)

        with open(fn_save, 'wb') as f:
            pickle.dump(results_dict, f)
            
        t2 = time.time()    
        print('seed {0} finished in {1} minutes'.format(s, (t2-t1)/60))
