import numpy as np
import pandas as pd
import os
import pickle
import argparse 
import scipy.sparse
import sklearn.metrics
import time

import train_fxns_nonimage as t
from dataset_params import dataset_params
from fit_scaling_law import get_group_fits, suggest_alpha
from subsetting_exp_nonimage import subset_and_train, kwargs_to_kwargs_specifier
from subsetting_exp_nonimage import fracs_to_subset_sizes_additional


data_dir = '../../data'
results_subset_path = '../../results/subset_results/'
results_pilot_path = '../../results/pilot_results/'

def split_pilot_additional(full_data, 
                           group_key, 
                           groups,
                           group_sizes_pilot,
                           seed,
                           fold_key = 'fold', 
                           verbose = True):
    
    rs = np.random.RandomState(seed)
    # put group_sizes_pilot number of samples from each group in the pilot sample;
    # the test set goes into the eval data, and the rest into additional data
    data = full_data.copy()
    data_eval = data[data[fold_key] == 'test'].copy()
    if verbose:
        print(len(data_eval), ' pts in eval set')
    
    data_train = data[data[fold_key] != 'test'].copy()

    # split data_train by group, and add to pilot and additional 
    data_pilot = pd.DataFrame()
    data_additional = pd.DataFrame()
    
    for g,group in enumerate(groups):
        data_this_group = data_train[data_train[group_key] == group]
        shuffled_idxs = rs.choice(len(data_this_group), 
                                  len(data_this_group),
                                  replace = False)
        
        pilot_idxs_this = shuffled_idxs[:group_sizes_pilot[g]]
        additional_idxs_this = shuffled_idxs[group_sizes_pilot[g]:]
        data_pilot = data_pilot.append(data_this_group.iloc[pilot_idxs_this], 
                          ignore_index=True)
        data_additional = data_additional.append(data_this_group.iloc[additional_idxs_this], 
                                            ignore_index=True)
        
    # shuffle the data so it isn't sorted by group
    data_pilot = data_pilot.sample(frac=1, replace=False, random_state=rs).reset_index(drop=True)
    data_additional = data_additional.sample(frac=1, replace=False, random_state=rs).reset_index(drop=True)
    
    if verbose:
        print('of the remaining {0} pts: '.format(len(data_train)), end = ' ') 
        print('{0} pts in pilot, and {1} pts in additional'.format(len(data_pilot),
                                                               len(data_additional)))
    
    return data_pilot, data_additional, data_eval
    
    
# get all the alphas we want to use for the subsetting experiment.
def alphas_to_all_subset_sizes(alphas, 
                               n_train_per_group, 
                               fracs_smaller = [0.0625,0.125,0.25,0.5,1.0],
                              #fracs_smaller = [1.0],
                               include_maxed = False,
                               include_equal_sizes = False,
                               min_pts=1):
    
    # do this in terms of alphas in case we have unequal n_train_per_group
    alphas_all = []
    for frac_smaller in fracs_smaller:
        if frac_smaller == 1.0:
            alphas_this = fracs_to_subset_sizes_additional(frac_smaller,alphas)
        else:
            # don't repeat the first alpha becaues it's so small
            alphas_this = fracs_to_subset_sizes_additional(frac_smaller,alphas[1:])
        alphas_all.append(alphas_this)
            
    if include_maxed:
        alphas_all.append(np.vstack((alphas, np.ones(len(alphas)))))
        alphas_all.append(np.vstack((np.ones(len(alphas)), alphas)))
        
    if include_equal_sizes:
        alphas_all.append((np.vstack((alphas, 1-alphas))))
        alphas_all.append((np.vstack((1-alphas, alphas))))
        
    # aggregate in one aray
    alphas_all = np.hstack(alphas_all)
    # transform to subset sizes
    subset_sizes_all = np.round(alphas_all * np.array(n_train_per_group).reshape(2,1))
    # don't repeat any (n_A, n_B) combos
    subset_sizes_all = np.unique(subset_sizes_all, axis=1).astype(int)
    # if we won't use the subset size for either of the scaling fits, don't run it
    print('subsets would have run if min_pts: {0}'.format(subset_sizes_all.shape[1]))
    subset_sizes_all = subset_sizes_all[:,(subset_sizes_all>= min_pts).any(axis=0)]
    print('total # subsets to run: {0}'.format(subset_sizes_all.shape[1]))
    return subset_sizes_all
    
    
def pilot_sample_experiment(seed,
                            full_data, 
                            X,
                            alphas, 
                            group_key,
                            groups,
                            gammas,
                            group_sizes_pilot,
                            n_news,
                            label_colname,
                            pred_fxn_name,
                            pred_fxn, 
                            model_kwargs,
                            acc_fxns,
                            acc_key,
                            results_descriptor,
                            augment_original_data = False,
                            num_seeds_data_collection = 10,
                            start_seed_data_collection = 0,
                            fold_key = 'fold',
                            fit_obj='weighted_avg',
                            optimizer='max',
                            min_pts_scaling = 1,
                            scaling_data_saved_already=False
                           ):
    
    # 1. split into [pilot, additional, eval] data. Eval data will be the test set 
    #    that is already set aside
    pilot_data, additional_data, eval_data = split_pilot_additional(full_data,
                                                                    group_key,
                                                                    groups,
                                                                    group_sizes_pilot,
                                                                    seed,
                                                                    fold_key=fold_key)
     
    
    # 2. collect scaling law data using "pilot", and the specified params
    print('collecting data from which to fit scaling laws...')
    # 2a. collect and save the scaling law data
    data_pilot_train_and_eval = pd.concat((pilot_data, eval_data), ignore_index=True)
    
    # save the data for future reference
    data_dir_pilot_datasets = os.path.join(data_dir,'int_datasets_pilot')
    data_dir_this_pilots = os.path.join(data_dir_pilot_datasets, results_descriptor)
    
    for p in [data_dir_pilot_datasets, data_dir_this_pilots]:
        if not os.path.exists(p):
            print(p,' did not exist; making it now')
            os.makedirs(p)
    pilot_data.to_csv(os.path.join(data_dir_this_pilots,
                                   'pilot_data_seed_{0}.csv'.format(seed)))
    additional_data.to_csv(os.path.join(data_dir_this_pilots,
                                        'additional_data_seed_{0}.csv'.format(seed)))
    eval_data.to_csv(os.path.join(data_dir_this_pilots,
                                   'eval_data_seed_{0}.csv'.format(seed)))
      
    # instantiate fps where the output data from subsets is to be stored
    pred_fxn_base_name = 'subset_{0}'.format(group_key)
    this_results_path = os.path.join(results_subset_path, results_descriptor)
    
    this_results_path_pre = os.path.join(this_results_path,pred_fxn_base_name)
    results_path_this_pred_fxn = os.path.join(this_results_path,pred_fxn_base_name, pred_fxn_name)
    subset_fps_to_make  = [results_subset_path,
                    this_results_path,
                    this_results_path_pre,
                    results_path_this_pred_fxn]
    
    # instantiate fps where the output data from the whole experiment is to be stored
    this_results_path_pilot = os.path.join(results_pilot_path, results_descriptor)
    this_results_path_pilot_pre = os.path.join(this_results_path,pred_fxn_base_name)
    results_path_this_pilot_pred_fxn = os.path.join(this_results_path_pilot,pred_fxn_base_name, pred_fxn_name)

    pilot_fps_to_make = [results_pilot_path, 
                         this_results_path_pilot, 
                         this_results_path_pilot_pre, 
                         results_path_this_pilot_pred_fxn]
    
    for p in subset_fps_to_make+ pilot_fps_to_make:
        if not os.path.exists(p):
            print(p,' did not exist; making it now')
            os.makedirs(p)
            
    
    kwargs_specifier = kwargs_to_kwargs_specifier(model_kwargs)
    
    fn_data_collection_save = os.path.join(results_path_this_pred_fxn,
                                           kwargs_specifier+'_pilot_seed_{0}.pkl'.format(seed))
    
    fn_pilot_experiment = os.path.join(results_path_this_pilot_pred_fxn,
                                       kwargs_specifier+'_pilot_eval_results_seed_{0}.pkl'.format(seed))
        
        
    if not scaling_data_saved_already:
        # go from alphas to subset_sizes_subsetting.
        subset_sizes_to_run = alphas_to_all_subset_sizes(alphas, 
                                                         group_sizes_pilot,
                                                         include_maxed = False,
                                                         include_equal_sizes = False,
                                                          min_pts = min_pts_scaling)
    
        print('scaling data not saved; running subsets now for {0} seeds'.format(num_seeds_data_collection))
        # run the data
        accs_by_group, accs_total = subset_and_train(data_pilot_train_and_eval,
                                                     X,
                                                     group_key,
                                                     label_colname,
                                                     subset_sizes_to_run, 
                                                     pred_fxn, 
                                                     model_kwargs,
                                                     acc_fxns,
                                                     reweight=False,
                                                     reweight_target_dist=False,
                                                     fold_key = 'fold',
                                                     eval_key = 'test',
                                                     seed_start = start_seed_data_collection,
                                                     num_seeds = num_seeds_data_collection,
                                                     verbose = False)

        # aggregate results for saving
        results_dict = {'group_key': group_key,
                        'seed_start': start_seed_data_collection,
                        'num_seeds': num_seeds_data_collection,
                        'subset_sizes': subset_sizes_to_run,
                        'accs_by_group': accs_by_group, 
                        'accs_total': accs_total}

        # save the eval results for this hyperparameter setting
        print('saving results in ', fn_data_collection_save)
        with open(fn_data_collection_save, 'wb') as f:
                pickle.dump(results_dict, f)
    else:
        print('reading from ',fn_data_collection_save)
        with open(fn_data_collection_save, 'rb') as f:
            results_dict = pickle.load(f)
            
        accs_by_group = results_dict['accs_by_group']
        accs_total = results_dict['accs_total']
        subset_sizes_to_run = results_dict['subset_sizes']

        
    # 3. Fit scalings based on performance on "eval", and suggest alpha_hat from the scalings
    print('fitting scaling laws...')
    # 3a. (optional) load scaling law data from the previous experiments
    
    # 3b. Fit the parameters
    # fit scaling laws
    popts, pcovs = get_group_fits(groups,
                                  accs_by_group,
                                  subset_sizes_to_run, 
                                  min_pts = min_pts_scaling,
                                  acc_key = acc_key)
    
    
    
    eval_results_by_alpha = {}
    for j, n_new in enumerate(n_news):
        # 3c. suggest new alpha
        alpha_hat, f_hat, f_vals = suggest_alpha(n_new, gammas, popts,
                                                 obj=fit_obj,
                                                 optimizer=optimizer
                                                )
        print('suggested alpha_hat is {0}'.format(alpha_hat))


        # 4. Collect additional data and append points from "additional" to training set
        #    according to alpha in [gamma, 0.5, alpha_hat]

        alpha_equal_groups = 0.5
        alpha_compare = {'alpha_hat': alpha_hat, 
                         'gamma':gammas[0], 
                         'equal_alpha': alpha_equal_groups}

        alphas_try = [alpha_hat, gammas[0]]
        alpha_names = ['alpha_hat', 'gamma']
        if gammas[0] != alpha_equal_groups:
            alphas_try.append(alpha_equal_groups)
            alpha_names.append('equal_alpha')
        
        results_this_n = alpha_compare.copy()
        for i,alpha_new in enumerate(alphas_try):

            eval_results_this_alpha = {
                'alpha': alpha_new
            }

            print(alpha_new)
            print(n_new)
            print(int(alpha_new * n_new))

            # how many additional points to collect
            group_sizes_from_alpha_hat = [int(round(alpha_new * n_new)), 
                                          int(n_new- round(alpha_new * n_new))]
            group_sizes_from_alpha_hat = np.array(group_sizes_from_alpha_hat)

            if augment_original_data:
                group_sizes_additional = group_sizes_from_alpha_hat - np.array(group_sizes_pilot)
                print('want ',group_sizes_from_alpha_hat)
                if group_sizes_additional[0] <0:
                    print('cant collect negative samples')
                    group_sizes_additional = np.array([0, group_sizes_additional.sum()])
                if group_sizes_additional[1] <0:
                    print('cant collect negative samples')
                    group_sizes_additional = np.array([group_sizes_additional.sum(),0] )

                group_sizes_bigger = group_sizes_additional + np.array(group_sizes_pilot)

                print('adding ',group_sizes_additional)
                data_to_add, _, _ = split_pilot_additional(additional_data,
                                                     group_key,
                                                     groups,
                                                     group_sizes_additional,
                                                     seed)


                data_augmented_and_train = pd.concat((pilot_data,
                                                      eval_data, 
                                                      data_to_add)).sample(frac=1, 
                                                                           replace=False,                                                                    random_state=seed).reset_index(drop=True)
            else:
                print('picking a fresh ',group_sizes_from_alpha_hat, 'from additional data')
                data_to_add, _, _ = split_pilot_additional(additional_data,
                                                           group_key,
                                                           groups,
                                                           group_sizes_from_alpha_hat,
                                                           seed)

                data_augmented_and_train = pd.concat((data_to_add,
                                                      eval_data)).sample(frac=1, 
                                                                         replace=False,
                                                                         random_state=seed).reset_index(drop=True)

                group_sizes_bigger = group_sizes_from_alpha_hat

            data_augmented_and_train.to_csv(os.path.join(data_dir_this_pilots,
                                                         'augmented_data_{1}_seed_{0}.csv'.format(seed,
                                                                                                  alpha_names[i])))

            # 5. Evaluate results on the "eval" set for each 
            start_seed_data_evaluation = 0
            num_seeds_data_evaluation = 1
            # don't subset the training data for evaluation, but it does expect it to be 2d

            # we might not actually get to do our desired sample, so use group_sizes_bigger
            subset_sizes_eval = group_sizes_bigger.reshape(2,1)
            print(group_sizes_bigger)

            accs_by_group, accs_total = subset_and_train(data_augmented_and_train,
                                                         X,
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


            eval_results_this_alpha['accs_by_group'] = accs_by_group
            eval_results_this_alpha['accs_total'] = accs_total

            results_this_n[alpha_names[i]] = eval_results_this_alpha
        
        # append to the results for this n
        eval_results_by_alpha[n_new] = results_this_n
    
    # 6. save results
    results_dict = {'group_sizes_pilot':group_sizes_pilot,
               'n_news':n_news,
               'gamma': gammas[0],
               'equal_alpha': alpha_equal_groups,
               'eval_results_by_alpha': eval_results_by_alpha}
               
    # save the eval results for this hyperparameter setting
    fn_pilot_experiment_this = fn_pilot_experiment.replace('.pkl',
                                                           '_min_pts_{0}.pkl'.format(min_pts_scaling))
    print('saving results in ', fn_pilot_experiment_this)

    with open(fn_pilot_experiment_this, 'wb') as f:
        pickle.dump(results_dict, f)
        
    return results_dict
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, 
                        help='string identifier of the dataset')
    parser.add_argument('--num_seeds', type=int, default=10, 
                        help='number of seeds to run')
    parser.add_argument('--seed_beginning', type=int, default=0, 
                        help='seed to start with')
    parser.add_argument('--pred_fxn_name', type=str)
    
    args = parser.parse_args()
    dataset_name = args.dataset_name.lower()
    num_seeds, seed_beginning = args.num_seeds, args.seed_beginning
    pred_fxn_name = args.pred_fxn_name
    
    # set this to true if we've already collected data but e.g. need to change 
    # something about the scaling law fits
    scaling_data_saved_already = False
    
    # common across experiments for all datasets
    alphas_base = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, .3, .4,  0.5,  0.6, 0.7, 0.8, 0.9, 1.0])  
    
    start_seed_data_collection = 0
    n_seeds_data_collect = 5
    
    if dataset_name == 'goodreads':
      #  results_descriptor = 'goodreads_2k_ERM_debug',
        label_colname = 'rating'
        group_key = 'genre'
        group_keys_to_stratify_cv = [group_key]
        all_group_colnames = ['history', 'fantasy']

        data_dir_goodreads = os.path.join(data_dir, 'goodreads')
        data_fn = os.path.join(data_dir_goodreads,
                               'goodreads_{0}_{1}_5_{2}_fold_splits.csv'.format(all_group_colnames[0],
                                                                               all_group_colnames[1],
                                                                               group_key))

        features_fn =  data_fn.replace('5_{0}_fold_splits.csv'.format(group_key), 'features_2k.npz')
        

        data = pd.read_csv(data_fn)
        X_this = scipy.sparse.load_npz(features_fn)
        
        gamma0 = dataset_params['goodreads']['gamma']
        groups = [0,1]
        gammas = [gamma0, 1-gamma0]
        
        #pred_fxn_name = 'logistic_regression'
        
        if pred_fxn_name == 'logistic_regression':
            acc_fxns = {'mse': sklearn.metrics.mean_squared_error,
                        'mae': sklearn.metrics.mean_absolute_error}

            pred_fxn = t.fit_logistic_regression_multiclass
            model_kwargs = {'penalty': 'l2','C':1.0, 'solver':'lbfgs'}
            acc_key = 'mae'
        
        elif pred_fxn_name == 'ridge':
            acc_fxns = {'mse': sklearn.metrics.mean_squared_error,
                        'mae': sklearn.metrics.mean_absolute_error}

            pred_fxn = t.fit_ridge_regression
            model_kwargs = {'alpha': 0.1}
            acc_key = 'mse'

        # pick alpha to minimize the max over groups of MSE
        
        optimizer = 'min'
        fit_obj = 'max_over_groups'
        
        # params for the pilot study example
        total_train_per_group = 50000
        num_per_group_in_pilot = 2500
        group_sizes_pilot = [num_per_group_in_pilot,num_per_group_in_pilot]
        
        if num_per_group_in_pilot == 2500:
            n_news = [5000,10000,20000,40000]
        elif num_per_group_in_pilot == 5000:
            n_news = [10000,20000,40000]
        #n_new = total_train_per_group - num_per_group_in_pilot
        
        if pred_fxn_name == 'logistic_regression':
            min_pts_scaling = 250
           # min_pts_scaling = int(num_per_group_in_pilot/20) # pick from the fitting scaling laws results
        elif pred_fxn_name == 'ridge':
            min_pts_scaling = 1000
        
    ss_descriptor = 'additional'
    print(model_kwargs)
    results_descriptor = '{0}_pilot_ERM_{3}_{1}_{2}'.format(dataset_name, 
                                                            pred_fxn_name,
                                                            num_per_group_in_pilot,
                                                            ss_descriptor) 
    
    results_descriptor_save = '{0}_pilot_ERM_{4}_{1}_{2}_min_pts_{3}'.format(dataset_name, 
                                                                             pred_fxn_name,
                                                                             num_per_group_in_pilot,
                                                                             min_pts_scaling,
                                                                             ss_descriptor) 
    
    print('results descriptor will be ', results_descriptor_save)
    for seed in range(seed_beginning, seed_beginning+num_seeds):  
        t1 = time.time()
        # this will save intermediate files
        r = pilot_sample_experiment(seed,
                                      data, X_this.toarray(),
                                      alphas_base,
                                      group_key, groups,
                                      gammas,
                                      group_sizes_pilot,
                                      n_news,
                                      label_colname,
                                      pred_fxn_name = pred_fxn_name,
                                      pred_fxn = pred_fxn,
                                      model_kwargs = model_kwargs, 
                                      acc_fxns = acc_fxns,
                                      acc_key = acc_key,
                                      start_seed_data_collection= start_seed_data_collection,
                                      num_seeds_data_collection = n_seeds_data_collect,
                                      results_descriptor = results_descriptor,
                                      fit_obj=fit_obj,
                                      optimizer=optimizer,
                                      min_pts_scaling = min_pts_scaling,
                                      scaling_data_saved_already=scaling_data_saved_already)
        
        
        t2 = time.time()    
        print('seed {0} finished in {1} minutes'.format(seed, (t2-t1)/60))
    
   