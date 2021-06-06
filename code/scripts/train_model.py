import logging
import time

import sys

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def initialize_model(model_name, num_classes,use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    
    if model_name == "resnet50":
        """ Resnet50
        """
        model = models.resnet50(pretrained=use_pretrained)
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "resnet18":
        """ Resnet18
        """
        
        print('using resnet 18')
        model = models.resnet18(pretrained=use_pretrained)
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    else:
        print("Invalid model name, exiting...")
        exit()
        
    # send to gpu
    model = model.to(device)
    
    return model, input_size


# modified from group_DRO code: https://github.com/kohpangwei/group_DRO/blob/master/loss.py
def compute_group_avg(losses, group_idxs, n_groups):
        # compute observed counts and mean loss for each group
        group_map = (group_idxs == torch.arange(n_groups).unsqueeze(1).long().cuda()).float()
        group_counts = group_map.sum(1)
    #    print(group_counts)
        group_denoms = group_counts + (group_counts==0).float() # avoid nans
        
        # compute group losses as weighted average
        group_losses = (group_map @ losses.float().view(-1))/ group_denoms
        
        return group_losses, group_counts
    
        
def update_gdro_weights(batch_loss, group_weights, group_sizes, batch_group_ids, 
                        step_size=0.01, 
                        group_adjustments = 1.0,
                        verbose_gdro=False
                        ):
    
    # update the weights per group
    n_groups = len(group_weights)
    
    
    group_losses, batch_group_counts = compute_group_avg(batch_loss, batch_group_ids, n_groups)
    
    if verbose_gdro:
        print('batch losses', batch_loss)
    # include the adjustments -- assume for now each group has the same adjustement
    if group_adjustments  > 0.0:        
        group_adjs = torch.tensor(group_adjustments).to(device) / (torch.sqrt(group_sizes)+ (group_sizes==0).float()).to(device)
   #     print(group_adjs)
        group_losses += group_adjs
    
    # exponential update 
    group_weights = group_weights * torch.exp(step_size * group_losses.data)
       
    # renormalize   
    group_weights = group_weights / (group_weights.sum()) 
    
    if verbose_gdro:
        print('group losses ', group_losses)
        print('group weights ', group_weights)
        
    gdro_loss = group_losses @ group_weights
    
    return gdro_loss, group_weights, group_losses


def train_net(model, dataloaders, criterion, 
              optimizer, gdro=False, num_epochs=20, 
              verbose_gdro=False, 
              eval_each_epoch=False,
              **gdro_params):
    # based on fine tuning implementation from: 
    # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    phase = 'train'
    # train the model
    if phase == 'train':
        model.train()
    
    train_loader = dataloaders['train']
    if eval_each_epoch:
        val_loader = dataloaders['val']
    
    if gdro:
        # read gdro parameters
        num_groups = gdro_params['num_groups']
        group_sizes = gdro_params['group_sizes']
        group_adj = gdro_params['group_adjustment']
        group_key = gdro_params['group_key']
        gdro_step_size = gdro_params['gdro_step_size']
        
        # initialize weights to be the same for each group
        gdro_group_weights = torch.ones(num_groups).to(device) / num_groups
                                 
    t1 =  time.time()
    
    # Mock training loop
    for epoch in range(num_epochs):
        print(epoch)
        
        running_loss = 0.0
        running_corrects = 0
        
        if gdro:
            running_group_sizes = torch.tensor([0.0,0.0]).to(device)
            running_group_losses = torch.tensor([0.0,0.0]).to(device)
    
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        if gdro:
            print("weights before epoch: ", gdro_group_weights )
        
        for batch_idx, sample in enumerate(train_loader):
            
            images = sample['image']
            inputs = images.to(device)
            labels = sample['target'].to(device)
                     
            if gdro:
                group_ids = sample[group_key].to(device)
                group_sizes_this_batch = torch.tensor([torch.sum(group_ids == g) for g in range(num_groups)]).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)

                if gdro:
                    # if gdro is used, criterion ouputs per-instance loss
                    per_instance_loss = criterion(outputs, labels)
                    if verbose_gdro:
                        print("weights before: ", gdro_group_weights )
                        print("loss before: ", per_instance_loss )
                       
                    # calculate the loss as a weighted sum of group losses
                    loss, gdro_group_weights, group_losses = update_gdro_weights(per_instance_loss, 
                                                                   gdro_group_weights, 
                                                                   group_sizes,
                                                                   group_ids,
                                                                   step_size = gdro_step_size,
                                                                   group_adjustments = group_adj,
                                                                   verbose_gdro = verbose_gdro)
                    
                    if verbose_gdro:
                        print("weights after: ", gdro_group_weights )
                        print("loss after: ", loss )
            
                else:
                    # if gdro is not used, criterion ouputs average loss over the batch
                    loss =  criterion(outputs, labels)
                
            _, preds = torch.max(outputs, 1)

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        if gdro:
            # gdro statistics
            running_group_sizes += group_sizes_this_batch
            running_group_losses += group_losses.clone().detach() * group_sizes_this_batch
            
            print("per-group loss after epoch: ", running_group_losses / running_group_sizes)
            print("weights after epoch: ", gdro_group_weights )
            print("total loss after epoch: ", running_loss / torch.sum(running_group_sizes)) 


        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
        
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        
        if eval_each_epoch:
            
            running_loss_eval = 0.0
            running_corrects_eval = 0
            
            if gdro:
                running_group_losses_eval = torch.tensor([0.0,0.0]).to(device)
                running_group_corrects_eval = torch.tensor([0.0,0.0]).to(device)
                running_group_sizes_eval = torch.tensor([0.0,0.0]).to(device)
        
            phase = 'val'
            model.eval()
            
            for batch_idx, sample in enumerate(val_loader):
            
                images = sample['image']
                inputs = images.to(device)
                labels = sample['target'].to(device)
                
                if gdro:
                    group_ids = sample[group_key].to(device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    if gdro:
                        per_instance_loss = criterion(outputs, labels)
                        # calculate the loss as a weighted sum of group losses
                        val_loss, val_gdro_group_weights, val_group_losses = update_gdro_weights(per_instance_loss, 
                                                                   gdro_group_weights, 
                                                                   group_sizes,
                                                                   group_ids,
                                                                   step_size = gdro_step_size,
                                                                   group_adjustments = group_adj,
                                                                   verbose_gdro = verbose_gdro)
                        
                        val_group_counts = torch.tensor([torch.sum(group_ids == g) for g in range(num_groups)]).to(device)
                        
#                         print(group_losses)
#                         print(group_counts)
#                         print(group_losses * group_counts)
                        running_group_losses_eval += val_group_losses * val_group_counts
                        
                        val_group_corrects = torch.tensor([torch.sum(preds[group_ids == g] == labels.data[group_ids == g]) \
                                               for g in range(num_groups)]).to(device)
                        
                        
                        running_group_corrects_eval += val_group_corrects
                        running_group_sizes_eval += val_group_counts
                        
                    else:
                        loss =  criterion(outputs, labels)
                    
                    
                    running_loss_eval += loss.item() * inputs.size(0)
                    running_corrects_eval += torch.sum(preds == labels.data)
                    
            epoch_loss_eval = running_loss_eval / len(dataloaders[phase].dataset)
            epoch_acc_eval = running_corrects_eval.double() / len(dataloaders[phase].dataset)
                                                      
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss_eval, epoch_acc_eval))
            
            if gdro:
                epoch_group_loss_eval = running_group_losses_eval /  running_group_sizes_eval
                epoch_group_acc_eval = running_group_corrects_eval.double() / running_group_sizes_eval

                print('Per-group {} Loss: {} Acc: {}'.format(phase, epoch_group_loss_eval, epoch_group_acc_eval))
            
            # set back to train
            phase = 'train'
            model.train()
        
        time_elapsed = time.time() - t1
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    

def eval_model(model, dataloaders, criterion, eval_set = 'val', return_preds=False):
    # based on fine tuning implementation from: 
    # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    
    assert eval_set in ['val', 'test', 'train_eval']
        
    # eval mode
    model.eval()
    
    data_loader_eval = dataloaders[eval_set]
    
    t1 =  time.time()

    running_corrects = 0
    running_loss = 0

    if return_preds:
        preds_all = torch.zeros(len(data_loader_eval.dataset))
        labels_all = torch.zeros(len(data_loader_eval.dataset))
        
    for batch_idx, sample in enumerate(data_loader_eval):
        images = sample['image']
        inputs = images.to(device)
        labels = sample['target'].to(device)
            
        if batch_idx % 50 == 0:
            print('batch ',batch_idx)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        _, preds_binary = torch.max(outputs, 1)
        
        softmax = nn.Softmax(dim=1)
        preds_continuous = softmax(outputs)[:,1]

        if return_preds:
            preds_all[batch_idx * data_loader_eval.batch_size : \
                      (batch_idx + 1) * data_loader_eval.batch_size] = preds_continuous
            
            labels_all[batch_idx * data_loader_eval.batch_size : \
                      (batch_idx + 1) * data_loader_eval.batch_size] = labels

      
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds_binary == labels.data)

    epoch_loss = running_loss / len(data_loader_eval.dataset)
    epoch_acc = running_corrects.double() / len(data_loader_eval.dataset)
        
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(eval_set, epoch_loss, epoch_acc))
        
    time_elapsed = time.time() - t1
    print('Eval complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if return_preds:
        return preds_all, labels_all
        
