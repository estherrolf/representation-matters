import pandas as pd
import torch
import numpy as np
import sklearn.metrics


def get_labels_and_groups(data_loader_eval, group_key):

    groups_all = torch.zeros(len(data_loader_eval.dataset))
    labels_all = torch.zeros(len(data_loader_eval.dataset))

    for batch_idx, sample in enumerate(data_loader_eval):

        if batch_idx % 50 == 0:
            print('batch ',batch_idx)

        labels = sample['target']
        groups = sample[group_key]
        
        labels_all[batch_idx * data_loader_eval.batch_size : \
                   (batch_idx + 1) * data_loader_eval.batch_size] = labels

        groups_all[batch_idx * data_loader_eval.batch_size : \
                   (batch_idx + 1) * data_loader_eval.batch_size] = groups
        
    return {'labels':labels_all, 'groups': groups_all}

    
def compute_accs_by_group(results, group_key):
    results_by_group = results.groupby([group_key])

    # autopopulates to correct but it's really counting the total number in each group
    df_by_group = pd.DataFrame(results_by_group['correct'].count()).rename(columns={'correct':'count_total'})
    df_by_group['count_pos'] = results_by_group['label'].sum()
    df_by_group['TP'] = results_by_group['TP'].sum()
    df_by_group['TN'] = results_by_group['TN'].sum()
    df_by_group['FP'] = results_by_group['FP'].sum()
    df_by_group['FN'] = results_by_group['FN'].sum()
    df_by_group['acc'] = results_by_group['correct'].mean()
    
    def auc_roc_by_group(group):
        if len(np.unique(group['label'].values)) == 1:
            return np.nan
        
        return sklearn.metrics.roc_auc_score(group['label'], group['pred'])
    
    df_by_group['auc_roc'] = results_by_group.apply(auc_roc_by_group)
    
    return df_by_group

def grab_results_by_group(data_loader_eval, preds_continuous, group_key):
    # binary preds with 0.5 thresh
    preds = (preds_continuous > 0.5).int()
    label_dict = get_labels_and_groups(data_loader_eval, group_key)
    
    correct = (label_dict['labels'] == preds).int()
    incorrect = (label_dict['labels'] != preds).int()
    labels = label_dict['labels'].int()
    preds = preds.int()
    acc_df_all = pd.DataFrame({group_key: label_dict['groups'], 
                               'pred': preds_continuous.numpy(), 
                               'label': labels.numpy(),
                               'correct': correct.numpy(),
                               'TP': torch.mul(correct, labels),
                               'FP': torch.mul(incorrect, preds),
                               'TN': torch.mul(correct, preds.mul(-1).add(1)),
                               'FN': torch.mul(incorrect, labels),
                               'acc':  np.mean(correct.numpy()),
                               'auc_roc': sklearn.metrics.roc_auc_score(labels.numpy(), preds_continuous.numpy())
                            })
    
    acc_df_by_group = compute_accs_by_group(acc_df_all, group_key)
    
    
    return acc_df_by_group, acc_df_all

