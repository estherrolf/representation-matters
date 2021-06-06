import os
import numpy as np
import pickle
import itertools
import pandas as pd

def read_in_results(group_key,
                    results_type = 'subset',
                    results_identifier = 'isic_split_aug_debug',
                    num_seeds=1,
                    seed_start = 0,
                    obj='basic',
                    acc_keys = ['acc','auc_roc'],
                    sgd_params = {'lr': 0.001, 'weight_decay': 0.001, 'momentum': 0.9},
                    num_epochs = 20,
                    add_reverse_accs = True,
                    pilot_data = False,
                    pilot_data_phase = 'data_collection',
                    return_preds = False,
                    read_by_seed_ls = False,
                    seed_ls = []):
    
    
    if len(str(sgd_params['weight_decay']).split('_')) > 1:
        param_str = "lr_{0}_weight_decays_{1}_momentum_{2}_epochs_{3}"
    else:
        param_str = "lr_{0}_weight_decay_{1}_momentum_{2}_epochs_{3}"
    sgd_param_specifier = param_str.format(sgd_params['lr'],
                                           sgd_params['weight_decay'],
                                           sgd_params['momentum'],
                                           num_epochs)
    
    if pilot_data and pilot_data_phase == 'pilot_eval_results':
        results_type_descriptor = 'pilot'
    else:
        results_type_descriptor = results_type
    results_folder = '../../results/{0}_results/{1}'.format(results_type_descriptor,
                                                    results_identifier,
                                                   )
    # pilot data saved in different format
    if pilot_data:
        res_str = '{4}_{1}/{2}_{3}_' + pilot_data_phase + '_seed_{0}.pkl'
    else:
        res_str = '{4}_{1}_seed_{0}_{2}_{3}.pkl'
    
    if (obj == 'GDRO'):
        # subset_animal_seed_9_lr_0.001_weight_decay_0.0001_momentum_0.9_epochs_20_GDRO_group_adj_4.0_gdro_stepsize_0.01
        param_str = "lr_{0}_weight_decay_{1}_momentum_{2}_epochs_{3}_GDRO_group_adj_{4}_gdro_stepsize_{5}"
        sgd_param_specifier = param_str.format(sgd_params['lr'],
                                               sgd_params['weight_decay'],
                                               sgd_params['momentum'],
                                               num_epochs,
                                               sgd_params['GDRO_group_adj'],
                                               sgd_params['gdro_stepsize'])
        
        res_str = '{3}_{1}_seed_{0}_{2}.pkl'
        file_name = os.path.join(results_folder,
                                 res_str.format('{0}', 
                                                group_key,
                                                sgd_param_specifier,
                                                results_type))
    else:
        if (isinstance(sgd_params['weight_decay'], str)):
             param_str = "lr_{0}_weight_decays_{1}_momentum_{2}_epochs_{3}"
        else:
             param_str = "lr_{0}_weight_decay_{1}_momentum_{2}_epochs_{3}"
        sgd_param_specifier = param_str.format(sgd_params['lr'],
                                               sgd_params['weight_decay'],
                                               sgd_params['momentum'],
                                               num_epochs)
        res_str = '{4}_{1}_seed_{0}_{2}_{3}.pkl'
        file_name = os.path.join(results_folder,
                                 res_str.format('{0}', 
                                                group_key,
                                                sgd_param_specifier,
                                                obj,
                                                results_type))
    subset_results = []
    if read_by_seed_ls:
        for seed in seed_ls:
            with open(file_name.format(seed), 'rb') as f:
       #         print(f'looking in {file_name.format(seed)}')
                subset_results_this_seed = pickle.load(f)
                subset_results.append(subset_results_this_seed)
    else:
        for seed in range(seed_start, seed_start+ num_seeds):
            with open(file_name.format(seed), 'rb') as f:
       #         print(f'looking in {file_name.format(seed)}')
                subset_results_this_seed = pickle.load(f)
                subset_results.append(subset_results_this_seed)
    
            
    # eval by group
    subset_sizes = subset_results[0]['{0}_sizes'.format(results_type)].T

    subset_fracs = (subset_sizes.T / subset_sizes.sum(axis=1))

    groups = subset_results[0]['{0}_groups'.format(results_type)]
    num_groups = len(groups)

    accs_dict_per_group = {}                         
    accs_dict_all = {} 

    for acc_key in acc_keys:
        accs_dict_per_group[acc_key] = np.zeros((num_groups, subset_sizes.shape[1], num_seeds))
        accs_dict_all[acc_key] = np.zeros((subset_sizes.shape[1], num_seeds))


    for i in range(num_seeds):
        for j in range(subset_sizes.shape[1]):
            results_per_group_this_allocation = subset_results[i]['accs_by_group'][j]

            for g in range(num_groups):
                for acc_key in acc_keys:
                    r = results_per_group_this_allocation.iloc[g][acc_key]
                    accs_dict_per_group[acc_key][g,j,i] = r

            results_all_this_allocation = subset_results[i]['accs_total'][j]
            for acc_key in acc_keys:
                accs_dict_all[acc_key][j,i] = results_all_this_allocation[acc_key].mean()

    if add_reverse_accs:
        for acc_key in acc_keys:
            new_key  = '1 - {0}'.format(acc_key)
            accs_dict_all[new_key] = 1 - accs_dict_all[acc_key]
            accs_dict_per_group[new_key] = 1 - accs_dict_per_group[acc_key]
        
        
    if return_preds:
        preds = []
        for i in range(num_seeds):
            preds_this_seed = []
            for j in range(subset_sizes.shape[1]):
                results_all_this_allocation = subset_results[i]['accs_total'][j]
                preds_this_seed.append(results_all_this_allocation[[group_key, 'label', 'pred']])
            preds.append(preds_this_seed)
                                       
        return groups, subset_sizes, accs_dict_all, accs_dict_per_group, preds
    else:
        return groups, subset_sizes, accs_dict_all, accs_dict_per_group

def read_subset_results_nonimage(results_path_this_pred_fxn, 
                                 param_dict,
                                 alternative_to_param_specifier=None,
                                 by_seed = False,
                                 seed_start = 0,
                                 num_seeds = None,
                                 add_reverse_accs = True,
                                 acc_keys = ['acc','auc_roc']):

    if by_seed:
        assert not num_seeds is None, 'must have num_seeds to read results by seed'
    
    use_param_dict_to_loop = (alternative_to_param_specifier is None)
    if use_param_dict_to_loop:
        hp_param_keys = list(param_dict.keys())
        hps_to_loop = itertools.product(*list(param_dict.values()))
    else:
        hps_to_loop = [0]
    # fill in results
    rs = {}
    
    if by_seed:
        for hp_setting in hps_to_loop:
            for s in range(seed_start, seed_start + num_seeds):
                model_kwargs = {}
                if use_param_dict_to_loop:
                    for i, hp in enumerate(hp_setting):
                        model_kwargs[hp_param_keys[i]] = hp
                    kwargs_specifier = '_'.join(['_'.join([str(y) for y in x]) for x in model_kwargs.items()])
                else:
                    kwargs_specifier = alternative_to_param_specifier       
                    

                fn_this = os.path.join(results_path_this_pred_fxn, kwargs_specifier + '_seed_{0}'.format(s)) + '.pkl'
                
                with open(fn_this, 'rb') as f:
                    d_this = pickle.load(f)
                # initialize with the first seed
                if s == seed_start:
                    d_all = d_this.copy()
                # add in the other seeds
                else:
                    for acc_key in acc_keys:
                    #print(acc_key)
                        d_all['accs_by_group'][acc_key] = np.concatenate((d_all['accs_by_group'][acc_key],
                                                         d_this['accs_by_group'][acc_key]), axis=2)
                        d_all['accs_total'][acc_key] = np.concatenate((d_all['accs_total'][acc_key],
                                                       d_this['accs_total'][acc_key]), axis=1)
                        
            rs[hp_setting] = d_all
    
    else:
        for hp_setting in hps_to_loop:
            model_kwargs = {}
            for i, hp in enumerate(hp_setting):
                model_kwargs[hp_param_keys[i]] = hp

            kwargs_specifier = '_'.join(['_'.join([str(y) for y in x]) for x in model_kwargs.items()])
            fn_this = os.path.join(results_path_this_pred_fxn, kwargs_specifier) + '.pkl'
   #         print(fn_this)
            with open(fn_this, 'rb') as f:
                results_this_hp_config = pickle.load(f)

            rs[hp_setting] = results_this_hp_config
    
    if add_reverse_accs:
        if use_param_dict_to_loop:
          #  print(param_dict)
            hps_to_loop = rs.keys()
            for hp_setting in hps_to_loop:
                for acc_key in acc_keys:
                    new_key  = '1 - {0}'.format(acc_key)
                    rs[hp_setting]['accs_by_group'][new_key] = 1 - rs[hp_setting]['accs_by_group'][acc_key]
                    rs[hp_setting]['accs_total'][new_key] = 1 - rs[hp_setting]['accs_total'][acc_key]
       
    return  rs

def get_best_hp_results(results_path_this_pred_fxn,param_dict, 
                        acc_key = 'mae', groups=[0,1], gammas = [0.5,0.5]):
    
    rs_dict = read_subset_results_nonimage(results_path_this_pred_fxn,param_dict, acc_keys=[acc_key])
    results_by_hps = pd.DataFrame(columns=['group','subset_frac'])    
    hp_param_keys = list(param_dict.keys())

    for hp_setting in itertools.product(*list(param_dict.values())):
            model_kwargs = {}
            for i, hp in enumerate(hp_setting):
                model_kwargs[hp_param_keys[i]] = hp

            results = rs_dict[hp_setting]
            accs_dict_per_group, accs_dict_all = results['accs_by_group'], results['accs_total']

            subset_sizes_cv = results['subset_sizes']
            subset_fracs = (subset_sizes_cv/subset_sizes_cv.sum(0))[0]
            
            
            for g,group in enumerate(groups + ['gamma_avged','min_over_groups','max_over_groups', 'pop']):
                for i,subset_frac in enumerate(subset_fracs):
                    this_res_dict = {'group':group, 
                                     'subset_frac': subset_frac,
                                     }
                    
                    this_res_dict.update(model_kwargs)
                    if group == 'pop':
                        for acc_key in [acc_key]:
                            avg_acc = np.mean(accs_dict_all[acc_key][i])
                            this_res_dict[acc_key] = avg_acc
                            
                    elif group == 'gamma_avged':
                        for acc_key in [acc_key]:
                            avg_acc_by_group = [np.mean(accs_dict_per_group[acc_key][g][i]) \
                                                for g in groups]
                            avg_acc_weighted = np.sum(np.array(avg_acc_by_group)*np.array(gammas))
                            this_res_dict[acc_key] = avg_acc_weighted
                                                        
                    elif group == 'min_over_groups':
                        for acc_key in [acc_key]:
                            min_acc = np.min([np.mean(accs_dict_per_group[acc_key][g][i]) \
                                                for g in groups])
                            this_res_dict[acc_key] = min_acc
                            
                    elif group == 'max_over_groups':
                        for acc_key in [acc_key]:
                            max_acc = np.max([np.mean(accs_dict_per_group[acc_key][g][i]) \
                                                for g in groups])
                            this_res_dict[acc_key] = max_acc
                    
                    elif group in groups:
                        for acc_key in [acc_key]:
                            avg_acc = np.mean(accs_dict_per_group[acc_key][g][i])
                            this_res_dict[acc_key] = avg_acc
                    # print(this_res_dict)
                    results_by_hps = results_by_hps.append(this_res_dict, ignore_index=True)
                
    res_grouped = results_by_hps.groupby(['group','subset_frac'])
    
    acc_key_to_sel = {'mse': 'min', 'mae': 'min', 'auc_roc': 'max', 'acc': 'max'}
    selection = acc_key_to_sel[acc_key]
    
    if selection == 'min':
        idxs_best = res_grouped[acc_key].idxmin()
    if selection == 'max':
        idxs_best = res_grouped[acc_key].idxmax()

    # aggregate results of the optimal for each pair, along with their accuracies
    op_df = pd.DataFrame()
    for g, group in enumerate(groups + ['gamma_avged', 'pop', 'min_over_groups','max_over_groups']):
        for subset_frac in subset_fracs:
            idx_max = idxs_best[group,subset_frac]

            op_df =op_df.append(results_by_hps.iloc[idx_max])

    return op_df, results_by_hps

def combine_data_results(subset_sizes_each, 
                         accs_by_group_each, 
                         accs_total_each):
    
    n = len(accs_by_group_each)
    accs_by_group_all = {}
    accs_total_all = {}
    
    # check if we need to resize b/c unequal number of seeds
    acc_key_0 = list(accs_by_group_each[0].keys())[0]
    accs_by_group_each_this_key = [accs_by_group_each[i][acc_key_0] for i in range(n)]
    num_seeds_each = [a.shape[-1] for a in accs_by_group_each_this_key]

    if len(np.unique(num_seeds_each)) != 1:
        # tile the subset sizes
        subset_sizes_each_repeated = [np.tile(ss,num_seeds_each[i]) for i,ss in enumerate(subset_sizes_each)]
        subset_sizes_all = np.hstack(subset_sizes_each_repeated)

        for k in accs_by_group_each[0].keys():
            accs_by_group_each_this_key = []
            accs_total_each_this_key = []
            for i in range(n):
                # stack all the seeds one by one
                a = accs_by_group_each[i][k]
                accs_by_group_this_reshaped = np.vstack([a[i].T.reshape(-1) for i in range(a.shape[0])]) 
                accs_by_group_each_this_key.append(accs_by_group_this_reshaped)
                
                b = accs_total_each[i][k]
                accs_total_each_this_key.append(b.T.reshape(-1))
                
            accs_by_group_all[k] = np.hstack(accs_by_group_each_this_key)
            accs_total_all[k] = np.hstack(accs_total_each_this_key)   
            
    
    else:
        subset_sizes_all = np.hstack(subset_sizes_each)
        for k in accs_by_group_each[0].keys():
            accs_by_group_each_this_key = [accs_by_group_each[i][k] for i in range(n)]
            num_seeds_each = [a.shape[-1] for a in accs_by_group_each_this_key]
            accs_by_group_all[k] = np.hstack(accs_by_group_each_this_key)

            accs_total_each_this_key = [accs_total_each[i][k] for i in range(n)]
            accs_total_all[k] = np.vstack(accs_total_each_this_key)

    return subset_sizes_all, accs_by_group_all, accs_total_all

def flip_group_results(accs_by_group, subset_fracs, group_id_dict, gammas, group_names):  
    # flips group A and group b
    subset_fracs_new = np.array([subset_fracs[1], subset_fracs[0]])
    gammas_new = [gammas[1], gammas[0]]
    group_names_new = [group_names[1], group_names[0]]
    
    accs_by_group_new = {} #accs_by_group.deepcopy()
    for acc_key in accs_by_group.keys():
        accs_by_group_new[acc_key] = np.zeros(accs_by_group[acc_key].shape)
        for g, group in enumerate(group_names):
            accs_by_group_new[acc_key][g] = accs_by_group[acc_key][1-g].copy()
            
    group_id_dict_new = group_id_dict.copy()
    for g, group in enumerate(group_names):
        group_id_dict_new[g] = group_id_dict[1-g]
    
    return accs_by_group_new, subset_fracs_new, group_id_dict_new, gammas_new, group_names_new
            
    
