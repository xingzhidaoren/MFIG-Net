'''
author: Jun Wang 
copyright: Hangzhou City University
times:2025.1.13
'''
import torch.nn as nn
class MLPClassifier(nn.Module):
    def __init__(self, layers=[64, 128, 128],  num_classes=3, dropout=0.2, device='cpu'):
        super(MLPClassifier, self).__init__()
        self.feature_extractor = nn.Sequential()
       
        for in_features, out_features in zip(layers, layers[1:]):   
            self.feature_extractor.append(
                nn.Linear(in_features, out_features)
            )
            self.feature_extractor.append(
                nn.ReLU()
            )
        self.classfier = nn.Sequential(
            nn.AdaptiveAvgPool1d(layers[-1]),
            nn.Linear(layers[-1], layers[-1]),
            nn.ReLU(),
            nn.Linear(layers[-1], layers[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=layers[-1], out_features=num_classes)
        )
        self.to(device)
                
    def forward(self, x, return_feature = False):
        '''
        x: (batch, features)
        '''
        featrues = self.feature_extractor(x)
        logits = self.classfier(featrues)
        if return_feature:
            return featrues, logits
        return logits