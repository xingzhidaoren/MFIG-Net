'''
author: jun wang
copyright:hzcu
date:2025.05.20
'''
import numpy as np
import torch
import sklearn.metrics as skm
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import os.path as osp
import os

def start_evaluate(model, dataset, mask_r, categores, save_path, save_predictions=True, feature_names=None, device='cpu'):
    '''
    Runing test and calculating metrics for the classification tasks.     
    Inputs
    model: the deepomic-trained model to evaluate
    dataset: the dataset used to evaluate the model
    mask_r: masking ratio to simluate missing values
    categores: the category names of the classification task
    save_path: save evaluation results to the directory
    save_predictions: if set to True, saving ema coefficeints, individual coefficents and predicted labels
    feature_names: the features names saved the columns to the coefficents csv files
    Outputs: 
    metrics.csv, if set save_predictions to True: ema_coeffs.csv, individual_coeffs.csv, predictions.csv
             These csv files are saved to the save_path
    '''
    num_classes = len(categores)
    assert num_classes>=2, 'num_classes must ≥ 2!'
    os.makedirs(save_path, exist_ok=True)

    model.eval()
    batch_size= 512
    val_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1, persistent_workers=True, shuffle=True)
   
    feature_importance = F.softmax(torch.abs(model.FSM.ema), dim=0)

    pred_labels = []
    pred_probs = []
    pred_coeffs = []
    gt_labels = []
    
    for xs, ys in val_dataloader:
        xs = torch.tensor(xs, dtype=torch.float32).to(device)               
        ys = torch.tensor(ys, dtype=torch.long).to(device)
        masked_xs,_ = __weighted_random_masking__(xs, mask_r, feature_importance)
        with torch.no_grad():
            _, _, coeffs, out_scores = model(masked_xs, return_intermediate_info=True)
        
        pred_labels.append(torch.argmax(out_scores, dim=-1).detach().cpu().numpy())
        pred_probs.append(out_scores.detach().cpu().numpy())
        pred_coeffs.append(coeffs.detach().cpu().numpy())
        gt_labels.append(ys.detach().cpu().numpy())

    pred_labels = np.concatenate(pred_labels, axis=0)
    pred_probs = np.concatenate(pred_probs, axis=0)
    pred_coeffs = np.concatenate(pred_coeffs, axis=0)
    gt_labels = np.concatenate(gt_labels, axis=0)

    '''print('pred:', pred_labels[0:5])
    print('gt:', gt_labels[0:5])'''

    metrics = calc_metrics(gt_labels, pred_labels, pred_probs, num_classes)

    df = pd.DataFrame(metrics, index=['value', '95%CI_low', '95%CI_high']).T   
    
    df.to_csv(osp.join(save_path,'metrics.csv'))

    if save_predictions:      
        if feature_names is None:
            feature_names = ['feature_{o}'.format(i) for i in range(pred_coeffs.shape[1])]
        df = pd.DataFrame([model.FSM.ema.detach().cpu().numpy()], columns=feature_names)
        df.to_csv(osp.join(save_path,'ema_coeffs.csv'), index=False)

        df = pd.DataFrame(pred_coeffs, columns=feature_names)
        df.to_csv(osp.join(save_path,'individual_coeffs.csv'), index=False)

        predictions = np.concatenate([pred_probs, pred_labels[...,None], gt_labels[...,None]],axis=-1)

        df = pd.DataFrame(predictions, columns=categores.extend(['pred labels', 'gt labels']))

        df.to_csv(osp.join(save_path,'predictions.csv'),index=False)

    return metrics

def __weighted_random_masking__(x, gamma, w):
    """
    Simplified version: masks exactly gamma*D features per sample,
    with probability inversely proportional to importance (w).
    
    Args:
        x: Tensor of shape (B, D) - input features
        gamma: float in (0, 1) - fraction of features to mask
        w: Tensor of shape (B, D) or (D,) - importance scores
        
    Returns:
        x_masked: Tensor of shape (B, D) - masked features
        mask: Tensor of shape (B, D) - binary mask (0=keep, 1=mask)
    """
    mask = torch.zeros_like(x, dtype=torch.bool)  # one = mask
    if gamma==0:
        return x, mask
    
    B, D = x.shape
    device = x.device
    
    sorted_ind = torch.argsort(w, descending=True)        
    n_keep = int(0.2*D)#top features are biomarkers,should be kept
    
    # Broadcast w to shape (B, D) if needed
    if w.dim() == 1:
        w = w.unsqueeze(0).expand(B, -1)
    
    # Compute sampling weights (inverse of importance)
    weights = 1.0-w  # smaller weight = more important
    
    # Sample k indices to mask per sample (lower weight = more likely to be chosen)
    
    for i in range(B):
        k = int(gamma * D)  # number of features to mask per sample
        if k==0:
            continue
        # Randomly select k indices based on weights
        selected = torch.multinomial(weights[i], k, replacement=False)
        mask[i, selected] = 1  # 1 = mask

        #number of kept vital variables         
        mask[i,sorted_ind[0:n_keep]]= 0
    
    # Apply mask
    x_masked = torch.where(mask==0, x, torch.tensor(-1, device=device))
    
    return x_masked, mask

def CI95_metrics_cls_task(pred_labels, pred_probs, ys, score_function, num_classes):
    '''
    pred_labels: the predicted labels
    pred_probs:  the predicted probablities
    ys: the ground-truth label
    score_function: the function used to calculate metrics
    '''
    alpha = 0.05 
    n_bootstraps = 100 
    scores = []   

    inds = np.array([i for i in range(ys.shape[0])])

    for _ in range(n_bootstraps):
        resampled_inds = np.random.choice(inds, size=len(inds), replace=True)
        resampled_preds = pred_labels[resampled_inds]
        resampled_pred_probs = pred_probs[resampled_inds]
        resampled_ys = ys[resampled_inds]
        if score_function == skm.roc_auc_score:
            if num_classes>2:
                scores.append(score_function(label_binarize(resampled_ys,classes=[0, 1, 2]), 
                                             resampled_pred_probs, average='macro', multi_class='ovr'))
            else:
                scores.append(score_function(resampled_ys, 
                                             resampled_pred_probs[:,-1]))
        elif score_function == skm.accuracy_score:
            scores.append(score_function(resampled_ys, resampled_preds))
        else:
            scores.append(score_function(resampled_ys, resampled_preds, average='macro'))


    lower_bound = np.percentile(scores, (alpha / 2) * 100)
    upper_bound = np.percentile(scores, (1 - alpha / 2) * 100)
    return lower_bound, upper_bound


def calc_metrics(gt_labels, pred_labels, pred_probs, num_classes):
    '''
    gt_labels:the groud-truth label
    pred_labels:the predicted label by model
    pred_probs:the predicted probabilities by model
    num_classes: the number of categories of the task
    '''
    accuracy = skm.accuracy_score(gt_labels, pred_labels)
    acc_low, acc_high = CI95_metrics_cls_task(pred_labels, pred_probs, gt_labels, skm.accuracy_score, num_classes)
    print('accuracy:', accuracy, '95%CI:', acc_low, acc_high)

    precision = skm.precision_score(gt_labels, pred_labels, average='macro')
    pre_low, pre_high = CI95_metrics_cls_task(pred_labels, pred_probs, gt_labels, skm.precision_score, num_classes)
    print('precision:', precision, '95%CI:', pre_low, pre_high )

    f1score = skm.f1_score(gt_labels, pred_labels, average='macro')
    f1_low, f1_high = CI95_metrics_cls_task(pred_labels, pred_probs, gt_labels, skm.f1_score, num_classes)
    print('f1score:', f1score, '95%CI:', f1_low,f1_high )

    recall = skm.recall_score(gt_labels, pred_labels, average='macro')
    rec_low, rec_high = CI95_metrics_cls_task(pred_labels, pred_probs, gt_labels, skm.recall_score, num_classes)
    print('recall:', recall, '95%CI:', rec_low, rec_high)

    if num_classes == 2:
        auc = skm.roc_auc_score(gt_labels, pred_probs[:, -1])        
    else:
        auc = skm.roc_auc_score(label_binarize(gt_labels,classes=[0, 1, 2]), 
                                             pred_probs, average='macro', multi_class='ovr')
    
    auc_low, auc_high = CI95_metrics_cls_task(pred_labels, pred_probs, gt_labels, skm.roc_auc_score, num_classes)
    print('auc:', auc, '95%CI:', auc_low, auc_high )

    metrics = {
        'accuracy': (accuracy, acc_low, acc_high),
        'precision': (precision, pre_low, pre_high),
        'f1score': (f1score, f1_low, f1_high),
        'recall': (recall, rec_low, rec_high),
        'auc': (auc, auc_low, auc_high)
        }
    
    return metrics


