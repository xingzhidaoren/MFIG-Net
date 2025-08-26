'''
author: jun wang
copyright:hzcu
date:2025.06.12
'''
import torch
import pandas as pd
import numpy as np
import torch.utils
from datasets.csv_dataset import CsvDataset
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from models.classifiers import ClassifierNames
from models.mfig_net import MFIGNet
from trainers.classifier_trainers import ClassifierTrainer
import os.path as osp
from evaluate import start_evaluate, calc_metrics
import warnings
warnings.filterwarnings('ignore')

def build_fold_path(task, classifier, frm_gama, fold):
    return '{0}_{1}_gama{2}/fold{3}'.format(task, classifier, frm_gama, fold)


if __name__ == '__main__':
    #define hyperparameters
    classifier = ClassifierNames.transformer #option: transformer, kan, longformer, mamba, resnet18, resnet50
    frm_gama = 0.2#the ratio to masking

    categories = ['viral', 'fungal', 'bacterial']
    num_classes = len(categories)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  

    
    #loading data samples, ignoire these columns in csv
    ignores = ['allergic history',
               'hospitalization days',
               'left lung volume',
               'right lung volume',
               'total lung volume']
    
    data_file = ['datasets/PathoCls-Pneu.csv']
    dataset = CsvDataset(data_file,
                                label_column='label',
                                ignores=ignores,
                                max_norm=True, #maximum normalization
                                maxv_for_norm=None,
                                repeat_fews=False)    
   
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (traind_ids, val_ids) in enumerate(kfold.split(dataset)):
        print('Folds:{0}/{1}'.format(fold+1, 5))
        train_subset = Subset(dataset, traind_ids)
        val_subset = Subset(dataset, val_ids)        

        model = MFIGNet(dataset.getNumFeatures(),
                        frm_gama = frm_gama,
                        frm_hiden_features=64,
                        fill_v=-1,
                        fsm_hiden_features=64,
                        fsm_out_activation='tanh',
                        fsm_ema_alpha=0.98,
                        classifier_name = classifier, 
                        classifier_layers = [dataset.getNumFeatures(), 64, 64, 64, 64, 64, 64],
                        num_classes= num_classes,
                        device=device)
        
        #define trainer for model training
        trainer = ClassifierTrainer(model, optimizer='adamw', batch_size=512, epochs=100, 
                                    lr=0.0001, lr_decay_steps=100, use_focal_loss=True, device=device)
        
        ckpoint_save_path = 'checkpoints/'+ build_fold_path('pneu_cls', classifier, frm_gama, fold+1)

        trainer.start_train(train_subset, ckpoint_save_path, val_subset, 1)

        #start evaluation for model performance
        print('Start testing')
       
        model.load_state_dict(torch.load(osp.join(ckpoint_save_path, "best_weights.pth"), map_location=device), strict=True)
       

        for mask_r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            metric_save_path = 'results/' + build_fold_path('pneu_cls', classifier, frm_gama, fold+1) + '/maskr{0}'.format(mask_r)
            start_evaluate(model, val_subset, mask_r, categories, metric_save_path, True, dataset.feature_names, device)


    
    
    #