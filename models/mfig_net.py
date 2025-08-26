'''
author: Jun Wang 
copyright: Hangzhou City University
times:2025.1.13
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .classifiers import GetClassifier

class FRM(nn.Module):
    '''
    the biomarker preserved random masking module for simulating missing values and recover these values
    '''
    def __init__(self, num_variables, gama=0.2, hiden_features=64, fill_v=-1, device='cpu'):
        super(FRM, self).__init__()
        '''      
        num_variables:the number of variables for each sample  
        gama: the masking ratio
        fill_v: the value used to replace the missing values, default set to -1
        device:cpu or cuda
        '''
        assert gama>=0.0 and gama<=1.0, \
            'masking ratio must in range of [0.0, 1.0], but given ' + str(gama)
        self.num_variables = num_variables
        self.gama = gama
        self.fill_v= torch.tensor(fill_v, dtype=torch.float32).to(device)
        self.device = device       
          
        #the masking probability 
        self.feature_importance = torch.nn.Parameter(torch.ones([num_variables]), requires_grad=False).to(device)*1.0/num_variables

        #reconstruction missing values
        self.reconstruction = nn.Sequential(
            nn.Linear(num_variables, hiden_features),  
            nn.ReLU(),         
           
            nn.Linear(hiden_features, hiden_features),  
            nn.ReLU(), 
          
            nn.Linear(hiden_features, hiden_features),  
            nn.ReLU(),  
           
            nn.Linear(hiden_features, num_variables),
            nn.Sigmoid() 
          
        )


    def forward(self, variables, model_kownledge=None):
        '''
        variabels: variables tensor (batch_size, num_features) 
        model_kownledge: the model weights of each variable (feedback from the ema of FSM module)
        '''
        #dropout only activated on training stage
        if self.training:
            if model_kownledge is not None:                
                #model_kownledge = torch.where(torch.isnan(model_kownledge), 0.0, model_kownledge)
                
                self.feature_importance = F.softmax(torch.abs(model_kownledge), dim=0)
                #print(self.dropout_p)
                if torch.isnan(self.feature_importance).any():
                    print('feature_importance has nan')
                   
                if torch.isinf(self.feature_importance).any():
                    print('feature_importance has inf')               
            masked_variables, mask = self.__weighted_random_masking__(variables, self.gama, self.feature_importance)
            recon_variables = self.reconstruction(masked_variables)
            return_variables = torch.where(mask==1, recon_variables, variables)
            return return_variables, mask
        else:
            recon_variables = self.reconstruction(variables)
            return_variables = torch.where(variables==self.fill_v, recon_variables, variables)
            return return_variables


    def __weighted_random_masking__(self, x, gamma, w):
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
        n_keep = int(0.5*D)#top features are biomarkers,should be kept
        
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
        x_masked = torch.where(mask==0, x, torch.tensor(self.fill_v, device=device))
        
        return x_masked, mask
    
    
    
class FSM(nn.Module):
    def __init__(self, num_variables, hiden_features=64, out_activation='tanh', ema_alpha = 0.99, device='cpu'):
        super(FSM, self).__init__()
        self.num_variables = num_variables
        self.register_buffer('ema',torch.nn.Parameter(torch.zeros([num_variables]), requires_grad=True).to(device)*1.0 )
        self.ema_alpha= ema_alpha
       
        self.feature_selector = nn.Sequential(
           nn.Linear(num_variables, hiden_features),
           nn.ReLU(),

           nn.Linear(hiden_features, hiden_features),
           nn.ReLU(), 

           nn.Linear(hiden_features, hiden_features),
           nn.ReLU(), 

           nn.Linear(hiden_features, num_variables),
           self.__get_activation__(out_activation)
           )
        self.to(device)
        
    def __get_activation__(self, activation_name):
        if activation_name=='tanh':
            return nn.Tanh()
        elif activation_name == 'sigmoid':
            return nn.Sigmoid()
        elif activation_name == 'softmax':
            return nn.Softmax()
        else:
            NotImplemented

    def forward(self, variables):        
        correlation_coefficents = self.feature_selector(variables)
        selected_variables = variables*correlation_coefficents
        if self.training:
            #update ema 
            self.ema = self.ema_alpha * self.ema + (1-self.ema_alpha)*torch.mean(correlation_coefficents, dim=0)
          
        return correlation_coefficents, selected_variables


class MFIGNet(nn.Module):
    '''
    the deep model for end-to-end biomarker discovery and disease modeling
    '''
    def __init__(self, 
                 num_variables=120,
                 frm_gama=0.2,
                 frm_hiden_features= 64,
                 fill_v = -1,
                 fsm_hiden_features=64, 
                 fsm_out_activation= 'tanh',
                 fsm_ema_alpha = 0.99,
                 classifier_name='transformer',
                 classifier_layers = [64, 64, 64, 64, 64, 64],
                 num_classes=3,
                 device='cpu'):
        super(MFIGNet, self).__init__()
        '''
        num_variables: the number of variables 
        frm_gama: the masking ratio in FRM
        frm_hiden_features: the neurons of layers for reconstruction network in FRM
        fill_v: the value used to fill the missing values, default -1
        fsm_hiden_features: the hiden featrues for the subnetwork in the ANFS module. Note value 0 means not using this module
        fsm_out_activation: the activation used to get the coefficients, default tanh
        fsm_ema_alpha: the hyperparameter used to control the EMA of coefficients
        classifier_name: kan, mamba, or transformer that used for the classification task
        classifier_layers: used to control the depth and features of the classifier
        num_classes: the number of categories for the classification task       
        device: cpu or cuda
        '''

        self.device = device
      
        self.FRM = FRM(num_variables=num_variables, 
                               gama=frm_gama,
                               hiden_features = frm_hiden_features,
                               fill_v=fill_v,
                               device = device)
        
        self.FSM = FSM(num_variables=num_variables,
                               hiden_features=fsm_hiden_features,
                               out_activation=fsm_out_activation,
                               ema_alpha=fsm_ema_alpha,
                               device=device)

        self.classifier = GetClassifier(classifier_name, classifier_layers, num_classes, 0.2, device)        

        self.to(device)
        
    def forward(self, variables, return_intermediate_info = False):
        '''
        input: samples (N, num_variables)
        return_intermediate_info: if True return the intermediate information
        '''   
        frm_mask=None
        if self.training:
            frm_variables, frm_mask = self.FRM(variables, self.FSM.ema)
        else:
            frm_variables = self.FRM(variables, self.FSM.ema)

        fsm_coefficents, fsm_variables = self.FSM(frm_variables)
       
        outcome_logits = self.classifier(fsm_variables)
        outcome_scores = F.softmax(outcome_logits, dim=-1)

        if return_intermediate_info:             
            return frm_variables, frm_mask, fsm_coefficents, outcome_scores
        else:
            return outcome_scores
       

#unit testing of the DeepBioDis
if __name__ == '__main__':
    data = torch.rand(4,10)
    data[0,0] = -1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MFIGNet(10, 0.2, 64, -1, 64, 'tanh', 0.99, 'kan', [10, 64, 64, 64, 64, 64, 64], 3, device=device)
    
    data = data.to(device)

    model.train()
   
    frm_var, frm_mask, fsm_cof, outcome_scores = model(data, return_intermediate_info=True)
    print(frm_var)
    print(frm_mask)
   

