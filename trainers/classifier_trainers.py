import os
import time
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from focal_loss import FocalLoss
import torchmetrics
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import sklearn.metrics as skm
from trainers.base_trainer import BaseTrainer
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler

def GetOptimizer(optimizer_name, model, lr):
    if optimizer_name.lower()=='adamw':
        return torch.optim.AdamW(model.parameters(), lr = lr)
    elif optimizer_name.lower()=='adam':
        return torch.optim.Adam(model.parameters(), lr = lr)
    elif optimizer_name.lower()=='sgd':
        return torch.optim.SGD(model.parameters(), lr = lr) 
    elif optimizer_name.lower()=='rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr = lr) 
    else:
        return NotImplemented
    
def get_sampling_wights(dataset):
    # dataset: is an isntance of torch.utils.data.Subset
    labels = []
    for idx in dataset.indices:
        _, label = dataset.dataset[idx]
        labels.append(label)

   
    class_counts = Counter(labels)
   
    # 2. 为每个类别计算权重（通常使用 1 / class_count）
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

    # 3. 为每个样本分配权重
    sample_weights = [class_weights[label] for label in labels]

    # 4. 创建 WeightedRandomSampler
    # num_samples：你希望每个 epoch 采样多少样本（通常设为训练集大小）
    # replacement=True：允许重复采样（即少数类样本会被重复使用）
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),  # 每个 epoch 采样这么多样本
        replacement=True
    )
    return sampler
    
class ClassifierTrainer(BaseTrainer):
    def __init__(self, model,
                 optimizer='Adam', 
                 batch_size= 10,
                 epochs = 100,
                 lr=0.01, 
                 lr_decay_steps = 100,
                 lr_decay_rate = 0.98,   
                 use_focal_loss = True,              
                 device='cpu'):
        super(ClassifierTrainer, self).__init__(model, optimizer, lr, device)
        '''
        model: the model to train (two outputs for binary classification and multiclass crossentropy is used)
        optimizer: the optimizer used to train the model        
        batch_size: the number of samples per iteration
        epochs: the number of epoch to train the model
        lr: the learning rate
        lr_decay_steps: StepLR used to decay the lr every n steps. if set to 0, then no StepLR
        lr_decay_rate: StepLR rate used to control the lr
        use_focal_loss: if set to True, using focal loss, else using crossentroy loss
        '''
       
        self.batch_size=batch_size
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        self.epochs = epochs
        self.use_focal_loss = use_focal_loss
        

    def start_train(self, train_dataset, save_model_dir, val_dataset=None, max_ck_files_to_save=2, pretrained_file=None, repeat_fews=True):
        '''        
        train_dataset: pytorch dataset for training        
        save_model_dir: the path used to save checkpoints and training summaries
        val_dataset: pytorch dataset for online evaluation, default None. If None, 10% of train_dataset samples were randomly selected for test
        max_ck_files_to_save: the maximum number of checkpoint files to save (old wights will be auto deleted).
        pretrained_file: if specified, loading the pretrained weights to init the model.
        '''
       
        self._prepare_path(save_model_dir)
        self.model.to(self.device)

        if not (pretrained_file is None): self._load_pretrained(pretrained_file, self.device) 

        if repeat_fews:
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=2, 
                                      persistent_workers=True, sampler=get_sampling_wights(train_dataset), shuffle=False)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=2, 
                                      persistent_workers=True, shuffle=True)

        if val_dataset is not None:     
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=1, persistent_workers=True, shuffle=True)

        optimizer = GetOptimizer(self.optimizer, self.model, self.lr)
        if self.lr_decay_steps <=0: 
            scheduler = None
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size = self.lr_decay_steps, 
                                                    gamma=self.lr_decay_rate)  
        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
        
        df_summary = pd.DataFrame(columns = ['time', 'step', 'mse', 'loss', 'accuracy'])
        df_summary.to_csv(os.path.join(self.save_path, "training_summary.csv"), index=False)

        self.max_acc = 0
        self.steps_per_epoch = len(train_dataset)//self.batch_size
        if self.use_focal_loss:
            loss_func = FocalLoss(gamma=4.0)
        else:
            loss_func = torch.nn.CrossEntropyLoss()

        recon_loss_func = torch.nn.MSELoss()

        for epoch in range(1, self.epochs+1):
            print('\n################# epoch:'+str(epoch)+'/'+str(self.epochs))            

            t1 = time.time()

            loss, recon_loss = self.__train_epoch(train_dataloader, 
                                      optimizer,
                                      scheduler,
                                      grad_scaler,
                                      loss_func,
                                      recon_loss_func)
            
            if val_dataset is None:
                acc = -1
            else:            
                acc = self.__calc_accuracy(val_dataloader)            

            t2 = time.time()
            
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
           
            print ('\nloss: %.3f; Lr: %.6f; Used time (s): %.4f; recon_mse: %0.5f; accuracy: %.10f' %
                    (loss, current_lr, t2-t1, recon_loss, acc))
            
            #saving training information to csv file
            current_time = "%s" % datetime.now()  
            step = "Step[%d]" % epoch
            train_loss = '%f' % loss
            recon_loss_str = '%f' % recon_loss
            accuracy_str = '%f' % acc
           
            list = [current_time, step, recon_loss_str, train_loss, accuracy_str]
            df_summary = pd.DataFrame([list])
            df_summary.to_csv(os.path.join(self.save_path, "training_summary.csv"), mode='a', header=False, index=False)

            #only saving the best model            
            if self.max_acc < acc or acc==-1:
                self.max_acc = acc
               
                self.checkpoint_file = os.path.join(self.save_path, "best_weights.pth")  
                print ('Saving weights to %s' % (self.checkpoint_file))     
                self.model.eval()       
                torch.save(self.model.state_dict(), self.checkpoint_file)

            self._delete_old_weights(max_ck_files_to_save)
            
            
    def __train_epoch(self, train_dataloader, optimizer, scheduler, grad_scaler, loss_func, recon_loss_func):
        losses = []
        recon_losses = []
        self.model.train()
        for step, (xs, ys) in enumerate(train_dataloader):
            
            xs = torch.tensor(xs, dtype=torch.float32).to(self.device)
            ys = torch.tensor(ys, dtype=torch.long).to(self.device)
           
            with torch.cuda.amp.autocast(enabled=False):
                frm_var, frm_mask, fsm_cof, out_scores = self.model(xs, return_intermediate_info=True)
           
            loss = loss_func(out_scores, ys)  

            keep = torch.where(frm_mask==1)            
           
            if keep[0].numel() > 0:  
                recon_loss = recon_loss_func(xs[keep], frm_var[keep])
            else:                
                recon_loss = torch.tensor(0.0, device=xs.device, requires_grad=True)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss+recon_loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
           
            if scheduler: scheduler.step()
            
            losses.append(loss.detach().cpu().numpy())
            recon_losses.append(recon_loss.detach().cpu().numpy())

            self._draw_progress_bar(step+1, self.steps_per_epoch)

        return np.mean(losses), np.mean(recon_losses)
    
    def __calc_accuracy(self, eval_dataloader):
        print('\nRuninig evaluation...')
        self.model.eval()
        pred_labels = []
      
        gt_labels = []
    
        for xs, ys in eval_dataloader:
            xs = torch.tensor(xs, dtype=torch.float32).to(self.device)               
            ys = torch.tensor(ys, dtype=torch.long).to(self.device)
            with torch.no_grad():
                out_scores = self.model(xs, return_intermediate_info=False)
            
            pred_labels.append(torch.argmax(out_scores, dim=-1).detach().cpu().numpy())
            
            gt_labels.append(ys.detach().cpu().numpy())

        pred_labels = np.concatenate(pred_labels, axis=0)
        gt_labels = np.concatenate(gt_labels, axis=0)
        accuracy = skm.accuracy_score(gt_labels, pred_labels)    
        '''print('pred:', pred_labels[0:5])
        print('gt:', gt_labels[0:5])'''
        return accuracy


        



