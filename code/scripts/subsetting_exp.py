import torch
#
torch.multiprocessing.set_sharing_strategy('file_system')

import pandas as pd
import numpy as np
import sys
import os
import pickle
import argparse

# local imports
import train_model
import experiment_routines as experiments
from subsetting_exp_nonimage import fracs_to_subset_sizes_additional

# can add different 
from dataset import get_data_loaders 
DATA_DIR = '../../data'

def subset_experiment(num_seeds, 
                      seed_beginning, 
                      group_key,
                      label_colname,
                      original_data_csv,
                      image_fp,
                      all_group_colnames,
                      fracs_group_a,
                      eval_key,
                      dataset_name,
                      run_type = 'subsetting',
                      results_descriptor='debug',
                      objs = ['ERM','IS'],
                      num_epochs = 20,
                      adjust_epochs_according_to_n = False,
                      dataset_size_adjust_for_num_epochs=None,
                      sgd_params = {'lr': 0.001, 'weight_decay': 0.001, 'momentum': 0.9},
                      gdro_params = {'num_groups': 2, 'group_adjustment': 1, 'gdro_step_size':0.01},
                      model_name = 'resnet18',
                      data_dir = DATA_DIR,
                      use_diff_wds = False,
                      wd_frac_range=[0.02, 0.98],
                      ):
    
    """Run the subsetting experiment. Assumes subsetting is performed on a binary variable.

    Arguments:
    num_seeds (int) -- how many random trials to run
    seed_beginning (int) -- beginning seed #
    group_key (str) -- column identifier of the column on which to subset, should be a column name of
                        original_data_csv
    label_colname (str) -- column idetifier of the column to predict, should be a column name of 
                           original_data_csv
    original_data_csv (str) -- file path (relative to multi-acc/data) of the full data csv
    image_fp (str) -- file path (relative to multi-acc/data) of the images listed in the full data csv
    all_group_colnames (list of strs) -- all columns that we might want to subset analysis on later
    fracs_group_a (1d array of floats) -- fractions of the dataset to assign to group A.
    results_descriptor (str) - string to store and later identify results. Results from the experiment
                               will be stored in multi-acc/results/<results_descriptor>
    """
    gdro_params = gdro_params.copy()
    gdro_params['group_key'] = group_key;

    objs = [x.upper() for x in objs]
    for obj in objs:
        assert obj in ['ERM', 'IS', 'GDRO'], \
        'objective {0} not supported (options are[ ERM, IS, GDRO])'.format(obj)
    
    # subset_groups is all of the different study_ids
    original_data = pd.read_csv(os.path.join(data_dir,original_data_csv))
    #print(original_data.keys())
    
    # get subset groups
    subset_groups = [[x] for x in list(original_data[group_key].unique())]
    # sorts in place
    subset_groups.sort()
    # print('subset groups are ', subset_groups)
        
    # select training data
    training_data = original_data[original_data['fold'] == 'train']
    num_training_pts = len(training_data)
     
    if run_type == 'max_all_groups' or run_type == 'leave_one_group_out':
        # assumes all subset groups contain one value only
        group_sizes = [(training_data[group_key].values == g[0]).sum() for g in subset_groups]
        print('group_sizes are: ', group_sizes)
    else:
        # double zero index b/c we know there's just one key in the group
        num_group_a = (training_data[group_key] == subset_groups[0][0]).sum()

        total_dataset_size_to_use = min(num_group_a, num_training_pts - num_group_a)
    
        train_str = 'there are {0} group {2} training pts and {1} other training pts'
        print(train_str.format(num_group_a,
                               num_training_pts - num_group_a,
                               subset_groups[0]))

        train_str_2 = 'so we use a total of {0} pts per trial in our experiment'
        print(train_str_2.format(total_dataset_size_to_use))  
        print()

    subset_sizes = np.ones((len(fracs_group_a), len(subset_groups)))
    
    train_fxns = {'ERM': experiments.train_and_eval_basic,
                  'IS': experiments.train_and_eval_IS, 
                  'GDRO': experiments.train_and_eval_GDRO}
    wd_range = []
    if (use_diff_wds):
        wd_range = [x * total_dataset_size_to_use for x in wd_frac_range]
    
    for t in range(seed_beginning, seed_beginning + num_seeds):
        print('seed ',t)
        print()
        # u shaped experiment
        if run_type == 'subsetting':
            adjust_epochs_according_to_n = False
            for s in range(len(fracs_group_a)):
                # how many points to alocate to A
                subset_size_group_a = int(fracs_group_a[s] * total_dataset_size_to_use)

                # group A
                subset_sizes[s,0] = subset_size_group_a 
                # group B
                subset_sizes[s,1] = total_dataset_size_to_use - subset_size_group_a

        elif run_type == 'additional':
            fracs_both_groups = fracs_group_a
            adjust_epochs_according_to_n = True
           
            subset_sizes = np.round(total_dataset_size_to_use * fracs_both_groups)
            # remove any duplicates
            subset_sizes = np.unique(subset_sizes,axis=1).T
            subset_sizes = subset_sizes.astype(int)
            
        elif run_type == 'additional_equal_group_sizes':
            fracs_both_groups = fracs_group_a
            adjust_epochs_according_to_n = True
           
            subset_sizes = np.round(total_dataset_size_to_use * fracs_both_groups)
            # remove any duplicates
            subset_sizes = np.unique(subset_sizes,axis=1).T
            subset_sizes = subset_sizes.astype(int)
            
        elif run_type == 'max_all_groups':
            subset_sizes[0] = group_sizes
            print('subset sizes: ',subset_sizes)
            
        elif run_type == 'leave_one_group_out':
            # full group sizes n_groups + 1 times
            subset_sizes = np.vstack([group_sizes]*(len(group_sizes)+1))
            # leave one out (the last one will max all groups)
            for i in range(len(group_sizes)):
                subset_sizes[i,i] = 0
                        
        else:
            print('run type {0} not understood'.format(run_type))
            return
            
        # data dict will store all the results
        data_dict = {}
        data_dict['subset_groups'] = subset_groups

        sgd_str = "lr_{0}_weight_decay_{1}_momentum_{2}_epochs_{3}"
        if (not use_diff_wds):
            sgd_param_specifier = sgd_str.format(sgd_params['lr'],
                                                 sgd_params['weight_decay'],
                                                 sgd_params['momentum'],
                                                 num_epochs)
        else:
            sgd_str = "lr_{0}_weight_decays_{1}_{2}_momentum_{3}_epochs_{4}"
            sgd_param_specifier = sgd_str.format(sgd_params['lr'],
                                                 sgd_params['weight_decay'][0],
                                                 sgd_params['weight_decay'][1],
                                                 sgd_params['momentum'],
                                                 num_epochs)
        
        
        results_general_path = '../../results/subset_results/'

        for obj in objs:            
            # set up results directory
            this_results_path = os.path.join(results_general_path, results_descriptor+'_'+obj)
            print('Will save results under:', results_descriptor+'_'+obj)
        
            for fp in [results_general_path,this_results_path]:
                if not os.path.exists(fp):
                    os.makedirs(fp)
            
            # 1.  run the experiment and save in fn_out for IS
            fn_out_str = os.path.join(this_results_path,
                                         'subset_{1}_seed_{0}_{2}_{3}.pkl')

            fn_out = fn_out_str.format(t,
                                       group_key,
                                       sgd_param_specifier,
                                       obj)
            
            if obj == 'GDRO':
                # add extra identifiers
                gdro_str = "group_adj_{0}_gdro_stepsize_{1}"
                gdro_param_specifier = gdro_str.format(gdro_params['group_adjustment'],
                                                       gdro_params['gdro_step_size'])
                fn_out = fn_out.replace('.pkl',
                                        '_{0}.pkl'.format(gdro_param_specifier))
                
            # second param here only matters if adjust_epochs_according_to_n = True
            train_fxn_params = {'num_epochs': num_epochs, 
                                'base_train_size_for_num_epochs': dataset_size_adjust_for_num_epochs}
            if obj != 'ERM':
                train_fxn_params['this_group_key'] = group_key
            
            tmp_data_descriptor = results_descriptor + obj + "_seed_{0}".format(t)

            experiments.subset_and_train(original_data_csv,
                                         image_fp,
                                         label_colname,
                                         all_group_colnames,
                                         group_key,
                                         subset_groups, 
                                         subset_sizes, 
                                         train_fxns[obj], 
                                         sgd_params,
                                         fn_out,
                                         eval_key,
                                         dataset_name,
                                         adjust_epochs_according_to_n = adjust_epochs_according_to_n,
                                         train_fxn_params = train_fxn_params,
                                         gdro = (obj == 'GDRO'),
                                         gdro_params = gdro_params,
                                         model_name= model_name,
                                         seed=t,
                                         tmp_data_descriptor=tmp_data_descriptor,
                                         use_diff_wds=use_diff_wds,
                                         wd_range=wd_range)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('group_key', type=str, 
                        help='which key to subset on')
    parser.add_argument('--num_seeds', type=int, default=10, 
                        help='number of seeds to run')
    parser.add_argument('--seed_beginning', type=int, default=0, 
                        help='seed to start with')
    parser.add_argument('--dataset_name', type=str, default='isic', 
                        help='seed to start with')
    parser.add_argument('--run_type', type=str, default='subsetting', 
                        help='subsetting or additional')
    parser.add_argument('--obj', type=str, default='ERM', 
                        help='string identifier of objective in [ERM, IS, GDRO]')
    parser.add_argument('--results_tag', type=str, default='debug', 
                        help='string identifier for the results folder that will be \
                        concatenated with the dataset name and objective')

    args = parser.parse_args()
    group_key = args.group_key
    num_seeds, seed_beginning = args.num_seeds, args.seed_beginning
    dataset_name = args.dataset_name
    run_type = args.run_type
    obj = args.obj
    results_tag = args.results_tag
    
    # default this to none
    dataset_size_base = None
    
    if run_type == 'subsetting':
        # change the line below to whatever it was
        results_descriptor = 'subsetting'
        if (obj == 'IS' or obj == 'GDRO'):
            fracs_group_a = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, .3, 0.35, .4, 0.45, 0.5, 0.55, \
                                      0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99])
        else:
            fracs_group_a = np.sort(np.append(np.linspace(0,1.0,num=21),[0.01, 0.02, 0.99, 0.98]))
        adust_epochs_according_to_n = False
        
    elif run_type == 'additional':
        results_descriptor = 'additional'
        adust_epochs_according_to_n = True
        
        # for debug:
        #fracs_group_a = np.array([0.2,0.5,1.0])
        # for full run:
        subset_fracs = []
        a_without_first = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, .3, .4,  0.5,  0.6, 0.7, 0.8, 0.9, 1.0])  
        for frac_smaller in [0.125,0.25,0.5]:
            subset_fracs.append(fracs_to_subset_sizes_additional(frac_smaller, a_without_first))
    
        # overload fracs group a with fracs from both groups
        fracs_group_a = np.hstack(subset_fracs)
        print('running additional')
        print(fracs_group_a.shape[1], ' trials per seed')
        
    elif run_type == 'additional_equal_group_sizes':
        results_descriptor = 'additional_equal_group_sizes'
        adust_epochs_according_to_n = True
        
        a = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, .3, .4,  0.5,  0.6, 0.7, 0.8, 0.9, 1.0])        
        fracs_group_a = fracs_to_subset_sizes_additional(1.0, a) 
        
        # overload fracs group a with fracs from both groups
        print('running additional with equal group sizes')
        print(fracs_group_a.shape[1], ' trials per seed')
    
    elif run_type == 'leave_one_group_out':
        print('running leave_one_group_out')
        results_descriptor = 'leave_one_group_out'
        adust_epochs_according_to_n = True
        # fracs_group_a won't get used
        fracs_group_a = [0.0]
        
    else:
        print('run type {0} not understood'.format(run_type))

    if dataset_name == 'isic':
        
        if (obj == 'ERM'):
            sgd_params = {'lr': 0.001, 'weight_decay': 0.0001, 'momentum': 0.9}
            gdro_params = {}
        elif (obj == 'IS'):
            sgd_params = {'lr': 0.001, 'weight_decay': 0.001, 'momentum': 0.9}
            gdro_params = {}
        elif (obj == 'GDRO'):
            sgd_params = {'lr': 0.001, 'weight_decay': 0.0001, 'momentum': 0.9}
            gdro_params = {'num_groups': 2, 'group_adjustment': 1.0, 'gdro_step_size': 0.1}
        else:
            print('Do not recognize objective')
        
        if adust_epochs_according_to_n:
            # size of training set on u-plot data
            dataset_size_base = 4092 
            
        subset_experiment(num_seeds, 
                          seed_beginning, 
                          group_key,
                          run_type = run_type,
                          label_colname = 'benign_malignant_01',
                          original_data_csv = 'isic/df_no_sonic_age_over_50_id.csv',
                          image_fp = 'isic/ImagesSmaller',
                          all_group_colnames = ['study_name_id',
                                                'age_approx_by_decade_id', 
                                                'age_over_45_id', 
                                                'age_over_50_id',
                                                'age_over_60_id', 
                                                'anatom_site_general_id', 
                                                'sex_id'],
                          objs= [obj],
                          fracs_group_a = fracs_group_a,
                          eval_key='test',
                          dataset_name='isic',
                          results_descriptor='isic_'+results_descriptor+'_'+results_tag,
                          num_epochs=20,
                          adjust_epochs_according_to_n = adust_epochs_according_to_n,
                          dataset_size_adjust_for_num_epochs=dataset_size_base,
                          sgd_params=sgd_params,
                          gdro_params=gdro_params
                          )
        
    elif dataset_name == 'isic_sonic':
        # this is a special case for the LOO expeirment
        if adust_epochs_according_to_n:
            print('adjusting epochs per n')
            # size of full training set
            dataset_size_base = 16965
            
        sgd_params = {'lr': 0.001, 'weight_decay': 0.001, 'momentum': 0.9}
        
        if group_key == 'study_name_aggregated_id':
            print('doing ',group_key)
            res_desc = 'isic_sonic_studies_aggregated'+results_descriptor
        else:
            res_desc = 'isic_sonic_'+results_descriptor
            
        subset_experiment(num_seeds, 
                          seed_beginning, 
                          group_key,
                          run_type = run_type,
                          label_colname = 'benign_malignant_01',
                          original_data_csv = 'isic/df_with_sonic_study_name_id.csv',
                          image_fp  = 'isic/ImagesSmallerWithSonic',
                          all_group_colnames = ['study_name_id',
                                                'study_name_aggregated_id',
                                                'age_approx_by_decade_id', 
                                                'age_over_45_id', 
                                                'age_over_50_id',
                                                'age_over_60_id', 
                                                'anatom_site_general_id', 
                                                'sex_id'],
                          objs= ['ERM'],
                          fracs_group_a = fracs_group_a,
                          eval_key='test',
                          dataset_name='isic_sonic',
                          results_descriptor=res_desc,
                          sgd_params = sgd_params,
                          num_epochs=20,
                          adjust_epochs_according_to_n = adust_epochs_according_to_n,
                          dataset_size_adjust_for_num_epochs=dataset_size_base,
                          )
        
    elif dataset_name == 'cifar':
        
        if (group_key == "air"):
            label_colname_ = "animal"
        else:
            label_colname_ = "air"
        
        if (obj == 'ERM'): 
            sgd_params = {'lr': 0.001, 'weight_decay': 0.0001, 'momentum': 0.9}
            gdro_params = {}
            use_diff_wds = False
            wd_frac_range = []
        elif (obj == 'IS'): 
            sgd_params = {'lr': 0.001, 'weight_decay': [0.01, 0.001], 'momentum': 0.9} 
            gdro_params = {}
            use_diff_wds = True
            wd_frac_range = [0.02, 0.98]
        elif (obj == 'GDRO'):
            sgd_params = {'lr': 0.001, 'weight_decay': 0.001, 'momentum': 0.9}
            gdro_params = {'num_groups': 2, 'group_adjustment': 4.0, 'gdro_step_size': 0.01}
            use_diff_wds = False
            wd_frac_range = []
        else:
            print('Do not recognize objective')
        
        if adust_epochs_according_to_n:
            # size of training set on u-plot data
            dataset_size_base = 10000
            
        subset_experiment(num_seeds,
                          seed_beginning, 
                          group_key,
                          run_type = run_type,
                          label_colname = label_colname_,
                          original_data_csv = 'cifar4/df_cifar4_labels.csv',
                          image_fp = 'cifar4/images',
                          all_group_colnames = ["air", "animal"],
                          fracs_group_a = fracs_group_a,
                          eval_key='test',
                          dataset_name='cifar',
                          results_descriptor = 'cifar4_'+results_descriptor+'_'+results_tag,
                          objs = [obj],
                          num_epochs=20,
                          adjust_epochs_according_to_n = adust_epochs_according_to_n,
                          dataset_size_adjust_for_num_epochs=dataset_size_base,
                          sgd_params=sgd_params,
                          gdro_params=gdro_params,
                          use_diff_wds=use_diff_wds,
                          wd_frac_range=wd_frac_range
                          )
    else:
        print('TODO: need to input the data and image files')
