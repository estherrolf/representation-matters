import matplotlib
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import fit_scaling_law
import matplotlib.font_manager

# FYI if tex is not installed, some of the plotting will break. 
def setup_fonts(use_tex=True):
    matplotlib.rcParams['pdf.fonttype'] = 42
    if use_tex: 
        plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"]})

    
def setup_uplot_ax():
    sns.set_context('talk')
    setup_fonts()
    
    figsize = (7,5)
    params = {'legend.fontsize': 'medium',
          'figure.figsize': figsize,
          'axes.labelsize': 'large',
          'axes.titlesize':'xx-large',
          'xtick.labelsize':'medium',
          'ytick.labelsize':'medium'}

    matplotlib.rcParams.update(params)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.01,1.01);
    
    return fig, ax

def setup_uplot_ax_smaller():
    sns.set_context('talk')
    setup_fonts()
    
    figsize = (7,4.3)
    params = {'legend.fontsize': 'medium',
          'figure.figsize': figsize,
          'axes.labelsize': 'medium',
          'axes.titlesize':'large',
          'xtick.labelsize':'medium',
          'ytick.labelsize':'medium'}

    matplotlib.rcParams.update(params)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.01,1.01);
    
    return fig, ax



def setup_scaling_plot_ax():
    sns.set_context('talk')
    setup_fonts()
    
    figsize = (7,5)
    params = {'legend.fontsize': 'large',
          'figure.figsize': figsize,
          'axes.labelsize': 'large',
          'axes.titlesize':'xx-large',
          'xtick.labelsize':'medium',
          'ytick.labelsize':'medium'}

    matplotlib.rcParams.update(params)
    fig, ax = plt.subplots(figsize=figsize)

    
    return fig, ax


def add_line(subset_fracs, accs_by_trial, color, label, ls, ax,
             lw = 2,
             label_range = None,
            range_type=None):
    # add line for upside down u-plots
    ax.plot(subset_fracs,
            np.mean(accs_by_trial,axis=1), 
            color=color,
            label=label, 
            lw = lw,
            ls = ls)
    

    n = accs_by_trial.shape[1]
    
    stddevs = np.std(accs_by_trial,axis=1)
    
    if range_type == 'stddev':
        ax.fill_between(subset_fracs,
                np.mean(accs_by_trial,axis=1) - stddevs,
                np.mean(accs_by_trial,axis=1) + stddevs,
                        label = label_range,
                color=color,
                alpha = 0.2)
        
    elif range_type == 'guassian_CI':
        ax.fill_between(subset_fracs,
                np.mean(accs_by_trial,axis=1) - 2 * stddevs / np.sqrt(n),
                np.mean(accs_by_trial,axis=1) + 2 * stddevs / np.sqrt(n),
                        label = label_range,
                color=color,
                alpha = 0.2)
        
    elif range_type == 'minmax':
        ax.fill_between(subset_fracs,
                np.min(accs_by_trial,axis=1),
                np.max(accs_by_trial,axis=1),
                        label= label_range,
                color=color,
                alpha = 0.2)
        
    elif not range_type is None:
        print('range specifier {0} not understood'.format(range_type))
    
    
def plot_by_group(accs_dict_per_group,
                  accs_dict_all,
                  subset_fracs,
                  acc_key,
                  group_id_dict,
                  gammas = [0.5,0.5],
                  pop_weights = 'by_eval_set',
                  colors = [sns.color_palette("colorblind")[0],
                            sns.color_palette("colorblind")[1], 
                            'black'],
                  label_append = '',
                  ls = '-',
                  lw = 3,
                  ax = None,
                  ylim= [0.8,1.0],
                  title='title me!',
                  range_type= 'stddev',
                  legend=True,
                  plot_gamma = False,
                  plot_alpha_star = False,
                  group_labels_only=False,
                  gamma_annot_offset = (0.0,0.0)):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    
    num_groups = accs_dict_per_group[acc_key].shape[0]

    if ls == '-':
        labels_range = [group_id_dict[g] for g in range(num_groups)] + ['population',]
    else:
        labels_range = [None]*3
        
    for g in range(num_groups):
        add_line(subset_fracs[0],
                 accs_dict_per_group[acc_key][g], 
                 color = colors[g],
                 label=None,
                 label_range = labels_range[g],
                 range_type = range_type,
                 ls = ls,
                 lw = lw,
                 ax=ax)

    if pop_weights == 'by_eval_set':        
        add_line(subset_fracs[0],
                 accs_dict_all[acc_key], 
                 color = colors[-1],
                 label= label_append,
                 label_range = labels_range[-1],
                 range_type = range_type,
                 ls = ls,
                 lw = lw,
                 ax=ax)
        
        accs_avgs = accs_dict_all[acc_key]
                
    elif pop_weights == 'by_gamma':
       
        accs_avgs = gammas[0]* accs_dict_per_group[acc_key][0] + \
                                  (1- gammas[0])* accs_dict_per_group[acc_key][1]
        add_line(subset_fracs[0],
                 accs_avgs, 
                 color = colors[-1],
                 #label= r'pop ($\gamma_A$ = {0:.2f})'.format(gammas[0],) + label_append,
                 label = label_append,
                 label_range = labels_range[-1],
                 range_type = range_type,
                 ls = ls,
                 lw = lw,
                 ax=ax)
        
                 
    else:
        print('cant assign pop weights by ',pop_weights)

    ax.set_xlabel(r'$\alpha_A$: frac {0} in training set'.format(group_id_dict[0]))
    
    ax.set_ylabel(acc_key)
    
    if plot_gamma: 
        gamma_color = 'dimgrey'
        ax.axvline(gammas[0], color=gamma_color, ls = '-', lw=2)
        
        ax.annotate(r'$\gamma_A = {0}$'.format(np.round(gammas[0],2)), (gammas[0]+ gamma_annot_offset[0], 
                                                            gamma_annot_offset[1]),
                    fontsize=24,
                    color=gamma_color)
        
    
    star_colors_by_ls = {'-':'white',
                         '--':'grey', 
                         ':':'black'}
    if plot_alpha_star:
        best_idxs_avged = accs_avgs.mean(axis=1).argmin()
        alpha_st_avged = subset_fracs[0][best_idxs_avged]
       
        ax.scatter([alpha_st_avged], [accs_avgs.mean(axis=1)[best_idxs_avged]],
                   marker='*', s=900, 
                   facecolors=star_colors_by_ls[ls], 
                   edgecolors='black',
                   lw=2,
                   zorder=4 + 1*(ls=='-'));
   
    if legend:
        # reorder so group labels are first
        handles, labels = ax.get_legend_handles_labels()
        num_objs = len(labels) - 3
        order = [num_objs + x for x in range(3)] + list(np.arange(num_objs))

        leg = ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                         ncol=2, fontsize=16)
        
        if group_labels_only:
            leg = ax.legend([handles[idx] for idx in order[:2]],[labels[idx] for idx in order[:2]],
                         ncol=1, fontsize=20, loc='upper center')
        
        # legend markers at maximum opacity
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
        
    if not ylim is None:
        ax.set_ylim(ylim[0],ylim[1])
    ax.grid(linewidth=1)
              
    ax.set_title(title)
    
def add_scatters(subset_fracs, accs_by_trial, plotting_kwargs, ax, label=None):
    # expand subset fracs for plotting
    subset_fracs_duplicated = np.tile(subset_fracs,(accs_by_trial.shape[1],1)).T
    ax.scatter(subset_fracs_duplicated.ravel(),
               accs_by_trial.ravel(), 
               label=label,
            **plotting_kwargs)
    
    return subset_fracs_duplicated.ravel(), accs_by_trial.ravel()


def plot_scaling_fits(subset_sizes,
                      accs_by_group, 
                      group_names,
                      acc_key,
                      popts,
                      n_thresh_for_scaling = 1,
                      n_thresh_for_plotting = 0,
                      colors = [sns.color_palette("colorblind")[0],
                            sns.color_palette("colorblind")[1], 
                            'black'],
                      ax = None,
                      loglog=True,
                      show_fitted_line = True,
                      show_data_not_fitted = True,
                      max_one_group = False,
                      full_legend=True,
                      dot_legend=False,
                      full_legend_loc_outside = False,
                      fit_fxn = fit_scaling_law.modified_ipl):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    
    group_sizes = []
    # this will differ per group if n_thresh_for_scaling != 0
    total_sizes = []
    
    # for legend
    dots = []
    est_fits = []
    extrap_fits = []
    for g, group in enumerate(group_names):
        
        
        subset_sizes_group_this = subset_sizes[g]
        
        if max_one_group: 
            g_other = 1 - g
            idxs_to_show = np.where(subset_sizes[g_other] == np.max(subset_sizes[g_other]))[0]
        else:
            idxs_to_show = np.arange(len(subset_sizes_group_this))
            
        in_range_idxs = np.where(subset_sizes_group_this[idxs_to_show] >= n_thresh_for_plotting)[0]
        idxs_to_show = idxs_to_show[in_range_idxs]

        x_g, y_g = fit_scaling_law.tile_data(subset_sizes_group_this[idxs_to_show],
                             accs_by_group[acc_key][g][idxs_to_show])
        
        
                                     
        subset_sizes_both_this = subset_sizes.sum(axis=0)[idxs_to_show]
        x_both, _ = fit_scaling_law.tile_data(subset_sizes_both_this, 
                             accs_by_group[acc_key][g][idxs_to_show])

        group_sizes.append(np.array(x_g))
        total_sizes.append(np.array(x_both))
        
        
            
        if show_data_not_fitted:
            # plot all as grey
            _,_  = add_scatters(subset_sizes[g][idxs_to_show],
                            accs_by_group[acc_key][g][idxs_to_show], 
                            plotting_kwargs={'edgecolor':'grey',
                                             'facecolor':'None',
                                             's':150},
                               ax=ax)

            ax.scatter(subset_sizes[g][idxs_to_show],
                       accs_by_group[acc_key][g][idxs_to_show].mean(axis=1), 
                            **{'color':'grey','s':200})
            
           
        idxs_mask_g = np.where(subset_sizes[g][idxs_to_show] >= n_thresh_for_scaling)
        _,_  = add_scatters(subset_sizes[g][idxs_to_show][idxs_mask_g],
                            accs_by_group[acc_key][g][idxs_to_show][idxs_mask_g], 
                     #       label=label_errors, 
                            plotting_kwargs={'edgecolor':colors[g],
                                             'facecolor':'None',
                                             's':150},
                               ax=ax)

        
        dots_this_group = ax.scatter(subset_sizes[g][idxs_to_show][idxs_mask_g],
                   accs_by_group[acc_key][g][idxs_to_show][idxs_mask_g].mean(axis=1), 
                   **{'color':colors[g],'s':200},
                   # priotize means on top
                  zorder=10)
        
        dots.append(dots_this_group)
            
        
    if show_fitted_line:
        # plot the fits
        for g, group in enumerate(group_names):
            # might need to argsort these if they're collected out of order
            subset_sizes_this = group_sizes[g]
            subset_sizes_total = total_sizes[g]
            xs_option = np.vstack((subset_sizes_this, 
                                   subset_sizes_total))

            min_pts_viz = n_thresh_for_plotting
            idxs_fit = np.where(subset_sizes_this >= n_thresh_for_scaling)[0]
            xs = xs_option[:,idxs_fit].T
            xs = xs[np.argsort(xs[:,0])]
            
            ax.plot(xs[:,0], 
                     [fit_fxn(x, *popts[g]) for x in xs], 
                     color=colors[g])


        # add dashed line
        if show_data_not_fitted:
            extrap_in_label = False
            for g, group in enumerate(group_names):
                
                subset_sizes_this = group_sizes[g]
                
                if (subset_sizes_this < n_thresh_for_scaling).sum() == 0:
                    continue
                
                in_range_idxs = np.where(subset_sizes_this >= n_thresh_for_plotting)[0]
         #       subset_sizes_total = total_sizes[g][in_range_idxs]
                xs_option = np.vstack((subset_sizes_this[in_range_idxs], 
                                       subset_sizes_total[in_range_idxs]))

                if max_one_group:
                    subset_sizes_other = subset_sizes_total - subset_sizes_this
                    idxs_group_maxed = np.where(subset_sizes_other == np.max(subset_sizes_other))[0]

                    xs_option = xs_option[:,idxs_group_maxed]

                min_pts_viz = n_thresh_for_plotting
                idxs_not_fitted = np.where(subset_sizes_this <= n_thresh_for_scaling)[0]
                idxs_geq1 = np.where(subset_sizes_this >=1)[0]
                idxs_this = np.intersect1d(idxs_not_fitted,idxs_geq1)
                xs = xs_option[:,idxs_this].T
                xs = xs[np.argsort(xs[:,0])]
                
                
                #if xs[:,0].max() < n_thresh_for_scaling:
                # find the next highest one
                idxs_fitted = np.where(subset_sizes_this > n_thresh_for_scaling)[0]
                ss_fitted = subset_sizes_this[idxs_fitted]
                idx_margin = idxs_fitted[ss_fitted.argmin()]
                    
                nj_margin, n_margin = subset_sizes_this[idx_margin], subset_sizes_total[idx_margin]
                xs = np.vstack((xs,[nj_margin, n_margin]))
                ax.plot(xs[:,0], 
                       [fit_fxn(x, *popts[g]) for x in xs], 
                       color=colors[g],
                       ls = '--')
                
                #  only label once
                if not extrap_in_label:
                    extrap_fit_this = ax.plot(xs[0,0], 
                            [fit_fxn(x, *popts[g]) for x in [xs[0]]], 
                            color='black',
                            ls = '--',
                            )
                    
                    extrap_fits.append(extrap_fit_this)
                    extrap_in_label = True
       
    
    #add legends
    means = ax.scatter([],[],  **{'color':'black','s':200})
    errs = ax.scatter([],[],**{'edgecolor':'black',
                                             'facecolor':'None',
                                             's':150})
    
    fits = [Line2D([0],[0], **{'color':'black'})]
    fit_names = ['estimated fits']
    
    if extrap_in_label:
        fits.append(Line2D([1],[1], **{'color':'black', 'ls':'--'}))
        fit_names.append('extrapolated fits')
                                 
    if full_legend and dot_legend and not full_legend_loc_outside:
        full_legend = ax.legend([means, errs] + fits + dots ,
                                ['mean errors' , 'per trial errors'] + fit_names + group_names,
                                 fontsize='small')
        ax.add_artist(full_legend)
    
    else:
        if full_legend:     
            if full_legend_loc_outside:
                full_legend = ax.legend([means, errs] + fits,['mean errors' , 'per trial errors'] + fit_names ,
                                        loc=(1.4,0.1),fontsize='xx-large', title='Legend')

                plt.setp(full_legend.get_title(),fontsize='xx-large')
            else:
                full_legend = ax.legend([means, errs] + fits,
                                        ['mean errors' , 'per trial errors'] + fit_names)

            ax.add_artist(full_legend)   

        if dot_legend:
            dot_legend = ax.legend(dots,group_names)
            ax.add_artist(dot_legend)
            
        
                    
    acc_key_to_print = {'1 - auc_roc': '1 - AUROC',
                        '1 - acc': '1 - accuracy',
                        'mae': 'MAE'}
    ax.set_ylabel(acc_key_to_print[acc_key])
        
    if loglog:
        ax.loglog()
            
    ax.grid()
    
    return ax

def trim_to_sigfigs(x, num_sigfigs = 2):
    
    if x < 1e-2 or  x >= 1e2:
        return '{0:.1e}'.format(x)
    return np.format_float_positional(x,precision=num_sigfigs, unique=False, fractional=False, trim='k')


def print_table_rows(dataset_name, group_names, popts, pcovs, min_pts_fit):

    specifier = '{0} ({1})'

    # collect components of the row
    popt_strings = []
    for g,group in enumerate(group_names):
        pieces = [group]
        for i, popt in enumerate(popts[g]):
            popt_ = trim_to_sigfigs(popt)
            pcov_ = trim_to_sigfigs(pcovs[g][i,i])
    
            pieces.append(specifier.format(popt_, pcov_))
        popt_strings.append(' & '.join(pieces))

    # format the row
    print_str = '\multirow{{2}}{{*}}{{{0}}} & '.format(dataset_name) + \
            '\multirow{{2}}{{*}}{{{0}}} & '.format(min_pts_fit) + \
            popt_strings[0] + ' \\\\ & & ' + popt_strings[1]
    
    print(print_str)
#   return print_str
