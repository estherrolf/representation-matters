import numpy as np

import scipy.optimize
from scipy.optimize import curve_fit


def modified_ipl(ns_stacked, sigmaj_sq, pj, tauj_sq, qj, deltaj):
    nj = ns_stacked[0]
    n = ns_stacked[1]
    
    
    return deltaj + sigmaj_sq * np.exp(-pj*np.log(nj)) + tauj_sq * np.exp(-qj*np.log(n)) 


def modified_ipl_logged(ns_stacked, sigmaj_sq, pj, tauj_sq, qj, deltaj):
    nj = ns_stacked[0]
    n = ns_stacked[1]
    
    
    return np.log(deltaj + sigmaj_sq * np.exp(-pj*np.log(nj)) + tauj_sq * np.exp(-qj*np.log(n)))


def modified_ipl_just_j(nj, sigmaj_sq, deltaj, pj):
    return sigmaj_sq * np.exp(-pj*np.log(nj)) + deltaj

def modified_ipl_logged_just_j(nj, sigmaj_sq, deltaj, pj):
    return np.log(sigmaj_sq * np.exp(-pj*np.log(nj)) + deltaj)

def get_group_fits(group_pair, 
                   accs_by_group, 
                   subset_sizes,
                   acc_key, 
                   min_pts = 1,
                   delta_bounds = [0,np.inf],
                   verbose=True, 
                   fit_logged=True,
                   need_to_tile_data=True):

    popts = []
    pcovs = []
    for g, group in enumerate(group_pair):
        if need_to_tile_data:
            ns, y = tile_data(subset_sizes.sum(axis=0), 
                              accs_by_group[acc_key][g])

            njs, _ = tile_data(subset_sizes[g], 
                              accs_by_group[acc_key][g])
        else:
            ns = subset_sizes.sum(axis=0)
            njs = subset_sizes[g]
            y = accs_by_group[acc_key][g]

        ns_input = np.vstack((njs, ns))
        popt, pcov = fit_scaling_law(ns_input, y, 
                                     delta_bounds = delta_bounds,
                                     min_pts = min_pts,
                                    fit_logged=fit_logged)

        popts.append(popt)
        pcovs.append(pcov)
        
        if verbose:
            print(group)
            keys = ['sigmaj_sq', 'p', 'tauj_sq', 'q', 'deltaj']
            for k, key in enumerate(keys):
                print("{0} : {1:.3f} ({2:.3f})".format(key,popt[k],np.sqrt(pcov[k,k])))
            print()
    return popts, pcovs

def get_group_fits_just_j(group_pair, 
                          accs_by_group, 
                          subset_sizes,
                          acc_key, 
                          
                          min_pts = 1,
                          verbose=True):

    popts = []
    pcovs = []
    for g, group in enumerate(group_pair):
        print(group)

        njs, y = tile_data(subset_sizes[g], 
                           accs_by_group[acc_key][g])

        popt, pcov = fit_scaling_law_just_j(njs, y, min_pts = min_pts)

        popts.append(popt)
        pcovs.append(pcov)
        
        if verbose:
            keys = ['sigmaj_sq', 'p', 'deltaj']
            for k, key in enumerate(keys):
                print("{0} : {1:.3f} ({2:.3f})".format(key,popt[k],np.sqrt(pcov[k,k])))
            print()
    return popts, pcovs

def fit_scaling_law(x,y , delta_bounds = [0,np.inf], fit_logged=True, min_pts = 1):
    
    bounds_ipl = np.array([[0,np.inf], [0,2], [0,np.inf], [0,2], delta_bounds]).T
    #bounds_ipl = np.array([[0,np.inf], [0,np.inf], [0,np.inf], [0,np.inf], delta_bounds]).T
    
    # scaling law won't work if there's zero of any data point
    valid_idxs = np.intersect1d(np.where(x[0,:] >=min_pts)[0],
                                np.where(x[1,:] >=min_pts)[0])
    
    # defaults to 1 unless otherwise stated
    initial_guess = [1,1,1,1,1e-8]
    
    if fit_logged:
        popt, pcov = curve_fit(modified_ipl_logged, 
                               x[:,valid_idxs], 
                               np.log(y[valid_idxs]),  
                               p0 = initial_guess,
                               bounds= bounds_ipl, 
                               max_nfev=1000)
        
    else:
        popt, pcov = curve_fit(modified_ipl,
                               x[:,valid_idxs], 
                               y[valid_idxs],  
                               p0 = initial_guess,
                               bounds= bounds_ipl, 
                               max_nfev=1000)
        
    # if we're hitting the bounds on exponents through a message
    if popt[1] == bounds_ipl[0,1] or popt[1] == bounds_ipl[1,1]:
        print('estimated pj hit bound: {0}'.format(popt[1]))
    if popt[3] == bounds_ipl[0,3] or popt[3] == bounds_ipl[1,3]:
        print('estimated qj hit bound: {0}'.format(popt[3]))
    
    return popt, pcov


def fit_scaling_law_just_j(x,y , delta_bounds = [0,np.inf], fit_logged=True, min_pts = 1):
    
    bounds_ipl = np.array([[0,np.inf], [0,1], delta_bounds]).T
    
    # scaling law won't work if there's zero of any data point
    valid_idxs = np.where(x >=min_pts)[0]
    
    
    # defaults to 1 unless otherwise stated
    initial_guess = [1,0.5,1e-8]
    
    if fit_logged:
        popt, pcov = curve_fit(modified_ipl_logged_just_j, 
                               x[valid_idxs], 
                               np.log(y[valid_idxs]),
                               p0 = initial_guess,
                               bounds= bounds_ipl, max_nfev=1000)
        
    else:
        popt, pcov = curve_fit(modified_ipl_just_j,
                               x[valid_idxs], y[valid_idxs],  
                               p0 = initial_guess,
                               bounds= bounds_ipl, max_nfev=1000)
        
        
    return popt, pcov

def tile_data(subset_sizes, errs_by_trial):
    subset_sizes_duplicated = np.tile(subset_sizes,(errs_by_trial.shape[1],1)).T
    return subset_sizes_duplicated.ravel(), errs_by_trial.ravel()




def suggest_alpha(n_new,  
                  gammas, 
                  popts, 
                  fit_fn=modified_ipl, 
                  obj='weighted_avg',
                  optimizer='max'):
    
    # all possible allocations
    alpha_candidates = np.linspace(0,1,n_new+1)
    na_candidates = np.linspace(0,n_new,n_new+1).astype(int)
    f_values_by_alpha = np.zeros(n_new+1)
    
    if obj == 'weighted_avg':
        def scaling_fxn(na, n):
            if na == 0 or n == 0:
                return np.inf
            
           

            f_value_0 = fit_fn([na, n], *popts[0])
            f_value_1 = fit_fn([n-na, n], *popts[1])

            return gammas[0] * f_value_0 + gammas[1] * f_value_1
        
    elif obj == 'min_over_groups':
        def scaling_fxn(na, n):
            if na == 0 or n == 0:
                return np.inf

            f_value_0 = fit_fn([na, n], *popts[0])
            f_value_1 = fit_fn([n-na, n], *popts[1])

            return gammas[0] * f_value_0 + gammas[1] * f_value_1
    
    elif obj == 'max_over_groups':
        def scaling_fxn(na, n):
            if na == 0 or n == na:
                return np.inf
            
            f_value_0 = fit_fn([na, n], *popts[0])
            f_value_1 = fit_fn([n-na, n], *popts[1])

            return np.max([f_value_0,f_value_1])
        
    elif obj == 'mix_over_groups':
        def scaling_fxn(na, n):
            if na == 0 or n == 0:
                return np.inf

            f_value_0 = fit_fn([na, n], *popts[0])
            f_value_1 = fit_fn([n-na, n], *popts[1])

            return np.min([f_value_0,f_value_1])
        
    f_values_by_alpha = [scaling_fxn(na, n_new) for na in na_candidates]

    if optimizer == 'max':
        best_idx = np.argmax(f_values_by_alpha)
    elif optimizer == 'min':
        best_idx = np.argmin(f_values_by_alpha)
        
    alpha_hat = alpha_candidates[best_idx]
    f_hat = f_values_by_alpha[best_idx]
    return alpha_hat, f_hat, f_values_by_alpha

