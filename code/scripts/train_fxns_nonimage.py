import numpy as np
import sklearn.metrics
from dataset_chunking_fxns import subsample_df_by_groups
import sklearn
import sklearn.linear_model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import time

# learning function: logistic regression multi-class
def fit_logistic_regression_multiclass(X,
                                       y,
                                       seed, 
                                       model_kwargs = {'penalty': 'l2', 'C':1}, 
                                       weights=None):
   
    if weights is None:
        weights = np.ones(len(y))
    else:
        weights = weights
        
        
    clf = sklearn.linear_model.LogisticRegression(**model_kwargs,
                                                  random_state = seed,
                                                  multi_class='multinomial',
                                                  max_iter=1000,
                                                  n_jobs = None)
    clf.fit(X, y, sample_weight=weights)
    return clf

# learning function: logistic regression
def fit_logistic_regression(X,
                            y,
                            seed, 
                            model_kwargs = {'penalty': 'l2', 'C':1}, 
                            weights=None):
    

    if weights is None:
        weights = np.ones(len(y))
    else:
        weights = weights
        
        
    clf = sklearn.linear_model.LogisticRegression(**model_kwargs,
                                                  random_state = seed,
                                                  multi_class='ovr',
                                                  max_iter=5000,
                                                  n_jobs = None)
    clf.fit(X, y, sample_weight=weights)

    return clf

def fit_rf_classifier(X, y, seed, 
                     model_kwargs = {'max_depth': None, 'n_estimators': 100}, 
                     weights=None):
    
    clf = RandomForestClassifier(**model_kwargs, random_state = seed, n_jobs=20)
    if  weights is None:
        weights = np.ones(y.shape)
        
    clf.fit(X, y, sample_weight=weights)
    return clf 

def fit_rf_regressor(X, y, seed, 
                     model_kwargs = {'max_depth': None, 'n_estimators': 100}, 
                     weights=None):
    clf = RandomForestRegressor(**model_kwargs, random_state = seed, n_jobs=20)
    
    if  weights is None:
        weights = 1.0

    clf.fit(X, y, sample_weight = weights)
        
    return clf 

def fit_ridge_regression(X, y, seed, 
                     model_kwargs = {'alpha': 1.0}, 
                     weights=None):
    reg = sklearn.linear_model.Ridge(**model_kwargs, normalize=True, random_state = seed, solver='svd')
    
    if weights is None:
        weights = np.ones(len(y))
    
    reg.fit(X, y, sample_weight=weights)
    return reg 

def subset_and_train(data,
                     features,
                     group_key,
                     label_key,
                     subset_sizes,
                     pred_fxn,
                     model_kwargs,
                     acc_fxns,
                     predict_prob=False,
                     reweight = False,
                     reweight_target_dist = None,
                     fold_key = 'fold',
                     eval_key='test',
                     seed_start = 0,
                     num_seeds = 5,
                     verbose=True):
    
    accs_total, accs_by_group = {}, {}
    for acc_key in acc_fxns.keys():
        accs_total[acc_key] = np.zeros((subset_sizes.shape[1],num_seeds))
        accs_by_group[acc_key] = np.zeros((2,subset_sizes.shape[1],num_seeds))
    
    groups = [[x] for x in range(subset_sizes.shape[0])]
    # run the training
    for s,seed in enumerate(range(seed_start,seed_start + num_seeds)):
        rs_this = np.random.RandomState(seed)

        print(seed,": ", end='')
        for i in range(subset_sizes.shape[1]):
            t1 = time.time()
            print(i, end = ' ')
            group_sizes_this = subset_sizes[:,i]
            if verbose:
                print(group_sizes_this, end = '')
            # subsample the dataset (training points only)
            data_subset = subsample_df_by_groups(data, 
                                                 group_key,
                                                 groups,
                                                 fold_key = fold_key,
                                                 group_sizes = group_sizes_this,
                                                 rs = rs_this,
                                                 keep_test_val = True, shuffle=True)

            data_subset_train = data_subset[data_subset[fold_key] == 'train']
            # eval on the following set
            data_subset_val = data_subset[data_subset[fold_key] == eval_key]
            # index into features
            train_idxs_this_round = data_subset_train['X_idxs']
            val_idxs_this_round = data_subset_val['X_idxs']
            
            X_train = features[train_idxs_this_round]
            X_val = features[val_idxs_this_round]
                 
            y_train, g_train = data_subset_train[label_key].values, data_subset_train[group_key].values
            y_val, g_val = data_subset_val[label_key].values, data_subset_val[group_key].values


            if reweight:
                # weights per group
                group_fracs_this = group_sizes_this / group_sizes_this.sum()
                train_weights_per_group = np.array(reweight_target_dist) / group_fracs_this
  #              print('train_weights_per_group ', train_weights_per_group)
#                print(train_weights_per_group)
                # weight per instance
                train_weights = np.array(train_weights_per_group)[g_train.astype(int)]
                # scale so that weights sum to n_train
                train_weights = len(train_weights) * train_weights / train_weights.sum()

            else: 
                train_weights = None
                
            # allow for passing in lists of model kwargs, in case HPs need to change with allocation
            if isinstance(model_kwargs, (list)):
                model_kwargs_this = model_kwargs[i]
                if verbose:
                    print(model_kwargs_this)
            else:
                model_kwargs_this = model_kwargs
                
            clf = pred_fxn(X_train, y_train, seed, 
                           weights=train_weights, model_kwargs=model_kwargs_this)
            
            if predict_prob:
                # take probability of class 1 as the prediction
                preds = clf.predict_proba(X_val)[:,1]
            else:
                preds = clf.predict(X_val)
            # if preds are already binary, this won't change anything
            rounded_preds = np.asarray([int(p > 0.5) for p in preds]) 
            
            
            for acc_key, acc_fxn in acc_fxns.items():
                if acc_key == 'acc':
                    accs_total[acc_key][i,s] = acc_fxn(y_val, rounded_preds)
                else:
                    accs_total[acc_key][i,s] = acc_fxn(y_val, preds)

            for g in range(2):
                for acc_key, acc_fxn in acc_fxns.items():
                    if acc_key == 'acc':
                        accs_by_group[acc_key][g,i,s] = acc_fxn(y_val[g_val == g], rounded_preds[g_val == g])
                    else:
                        accs_by_group[acc_key][g,i,s] = acc_fxn(y_val[g_val == g], preds[g_val == g])
            
            t2 = time.time()
            #print()
            if verbose:
                print('took {0} minutes'.format((t2-t1)/60))
        
        print()
        
    return accs_by_group, accs_total


def cv_subset_and_train(data, 
                        features,
                        group_key,
                        label_key,
                        subset_sizes,
                        pred_fxn,
                        model_kwargs,
                        acc_fxns,
                        predict_prob = False,
                        reweight=False,
                        reweight_target_dist=None,
                        num_seeds = 5,
                        verbose=True):

    accs_total, accs_by_group = {}, {}
    for acc_key in acc_fxns.keys():
        accs_total[acc_key] = np.zeros((subset_sizes.shape[1],num_seeds))
        accs_by_group[acc_key] = np.zeros((2,subset_sizes.shape[1],num_seeds))
    
        
    for seed in range(num_seeds):
        r = subset_and_train(data,
                             features,
                             group_key=group_key,
                             label_key=label_key,
                             subset_sizes=subset_sizes,
                             pred_fxn = pred_fxn,
                             model_kwargs = model_kwargs,
                             acc_fxns = acc_fxns,
                             reweight=reweight,
                             reweight_target_dist = reweight_target_dist,
                             predict_prob = predict_prob,
                             eval_key='val',
                             fold_key = 'cv_fold_{0}'.format(seed),
                             seed_start = seed,
                             num_seeds = 1,
                             verbose=verbose)
        
        accs_by_group_this_seed, accs_total_this_seed = r
        
        for acc_key in acc_fxns.keys():
            accs_total[acc_key][:,seed] = accs_total_this_seed[acc_key].reshape(-1)
            accs_by_group[acc_key][:,:,seed] = accs_by_group_this_seed[acc_key].reshape(2,-1)
                
    return accs_by_group, accs_total