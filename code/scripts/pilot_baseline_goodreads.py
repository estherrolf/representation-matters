import os

import pandas as pd
import numpy as np
import pickle

import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sklearn.linear_model

import sys
import argparse
sys.path.append('../../code/scripts')
sys.path.append('../scripts')

import train_fxns_nonimage as t
from dataset_chunking_fxns import subsample_df_by_groups
import plotting


import pilot_sample_experiment_nonimage as p
import train_fxns_nonimage as t
import pilot_sample_experiment_nonimage
from pilot_sample_experiment_nonimage import split_pilot_additional
from subsetting_exp_nonimage import subset_and_train
from dataset_params import dataset_params

import time
import scipy.sparse
from dataset_params import dataset_params


data_dir = '../../data'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('n_new', type=str, 
                        help='string identifier of the dataset')
    parser.add_argument('--num_seeds', type=int, default=10, 
                        help='number of seeds to run')
    parser.add_argument('--seed_beginning', type=int, default=0, 
                        help='seed to start with')
    
    #
    args = parser.parse_args()
    n_new = int(args.n_new)
    seed_start = args.seed_beginning
    num_seeds = args.num_seeds
    
    alphas = np.linspace(0,1.0,101)
    
    # get features
    all_group_colnames = ['history', 'fantasy']
    data_dir_goodreads = os.path.join(data_dir, 'goodreads')
    data_fn = os.path.join(data_dir_goodreads,
                                   'goodreads_{0}_{1}_5_fold_splits.csv'.format(all_group_colnames[0],
                                                                                   all_group_colnames[1]))

    features_fn =  data_fn.replace('5_fold_splits.csv', 'features_2k.npz')


    data = pd.read_csv(data_fn)
    X_this = scipy.sparse.load_npz(features_fn)
    
    acc_fxns = {'mse': sklearn.metrics.mean_squared_error,
                        'mae': sklearn.metrics.mean_absolute_error}

    pred_fxn = t.fit_logistic_regression_multiclass
    model_kwargs = {'penalty': 'l2','C':1.0, 'solver':'lbfgs'}
    acc_key = 'mae'


    group_key = 'genre'
    label_colname='rating'
    groups = [0,1]
    gamma0 = dataset_params['goodreads']['gamma']


    for seed in range(seed_start, seed_start+num_seeds):
        print('seed: {0}, n_new: {1}'.format(seed, n_new))
        eval_data_dir = '../../data/int_datasets_pilot/goodreads_pilot_ERM_additional_logistic_regression_2500/' + \
                        'eval_data_seed_{0}.csv'
        # the way we instantiated the eval set, these should all be the same 25k test set. Load them this way to 
        # maintain consistency with pilot_sample_
        eval_data = pd.read_csv(eval_data_dir.format(seed))

        additional_data_dir = '../../data/int_datasets_pilot/goodreads_pilot_ERM_additional_logistic_regression_2500/' + \
                        'additional_data_seed_{0}.csv'
        additional_data = pd.read_csv(additional_data_dir.format(seed))

        t1 = time.time()

        mae_by_group = np.zeros((len(alphas),4))
        for i,alpha in enumerate(alphas):

            group_sizes_from_alpha_hat = np.array([int(n_new*alpha), n_new - int(n_new*alpha)])

            print(i, ': picking a fresh ',group_sizes_from_alpha_hat, 'from additional data')
            data_to_add, _, _ = split_pilot_additional(additional_data,
                                                                   group_key,
                                                                   groups,
                                                                   group_sizes_from_alpha_hat,
                                                                   seed,
                                                      verbose=False)

            data_augmented_and_train = pd.concat((data_to_add,
                                                  eval_data)).sample(frac=1, 
                                                  replace=False,
                                                  random_state=seed).reset_index(drop=True)

            # don't actually subset futher, give the max number from each group for this alpha subset
            subset_sizes_eval = group_sizes_from_alpha_hat.reshape(2,1)

            # follow same routine as in pilot_sample_experiment_nonimage.py
            start_seed_data_evaluation = 0
            num_seeds_data_evaluation = 1

            accs_by_group, accs_total = subset_and_train(data_augmented_and_train,
                                                         X_this.toarray(),
                                                         group_key,
                                                         label_colname,
                                                         subset_sizes_eval, 
                                                         pred_fxn, 
                                                         model_kwargs,
                                                         acc_fxns,
                                                         reweight=False,
                                                         reweight_target_dist=False,
                                                         fold_key = 'fold',
                                                         eval_key = 'test',
                                                         seed_start = start_seed_data_evaluation,
                                                         num_seeds = num_seeds_data_evaluation,
                                                         verbose = False)

            mae_by_group[i,0] = accs_by_group['mae'][0,0,0]
            mae_by_group[i,1] = accs_by_group['mae'][1,0,0]
            mae_by_group[i,2] = gamma0*accs_by_group['mae'][0,0,0] + (1-gamma0)*accs_by_group['mae'][1,0,0]
            mae_by_group[i,3] = np.max((accs_by_group['mae'][0,0,0],accs_by_group['mae'][1,0,0]))
            
        t2 = time.time()
        print('seed {0} took {1:.2f} seconds'.format(seed, t2-t1))

        # save df
        df_this = pd.DataFrame({'alpha': alphas,
                      'mae_group_0': mae_by_group[:,0],
                      'mae_group_1': mae_by_group[:,1],
                      'mae_avg': mae_by_group[:,2],
                      'mae_max': mae_by_group[:,3],
                      'seed':[seed]*len(alphas)})

        results_dir = '../../results/pilot_results/goodreads_pilot_ERM_additional_logistic_regression_2500_bruteforce_baseline'
        csv_out_dir = os.path.join(results_dir,'baseline_additional_{1}_seed_{0}.csv'.format(seed,n_new))
        df_this.to_csv(csv_out_dir)