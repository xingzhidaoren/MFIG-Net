'''
author: jun wang
copyright:hzcu
date:2025.03.04
'''
from .kan import KANClassifier
from .nd_mamba import MambaClassifier
from .longformer import LongformerClassifier
from .resnets import resnet18, resnet34, resnet50, resnet101
from .MLP import MLPClassifier
from .transformer import TransformerClassifier

import torch

class ClassifierNames():
    kan = 'kan'
    mamba = 'mamba'
    transformer = 'transformer'
    resnet18 = 'resnet18'
    resnet34 = 'resnet34'
    resnet50 = 'resnet50'
    resnet101 = 'resnet101'   
    longformer = 'longformer'
    mlp ='mlp'

def GetClassifier(classifier, layers= [64, 128, 128], num_classes=3, dropout=0.2, device='cpu'):
    '''
    Return classifier with specified name and depth
    '''
    if classifier == 'kan':
        return KANClassifier(layers, num_classes, dropout, device)
    elif classifier == 'mamba':
        return MambaClassifier(layers, num_classes, dropout, device)
    elif classifier == 'transformer':
        return TransformerClassifier(layers, num_classes, 4, dropout, device)
    elif classifier == 'longformer':
        return LongformerClassifier(layers, num_classes, 4, 32, 256, dropout, device)
    elif classifier == 'mlp':
        return MLPClassifier(layers, num_classes, dropout, device)       
    elif classifier == 'resnet18':
        return resnet18(num_classes)
    elif classifier == 'resnet34':
        return resnet34(num_classes)
    elif classifier == 'resnet50':
        return resnet50(num_classes)
    elif classifier == 'resnet101':
        return resnet101(num_classes)
    else:
        NotImplemented    
     

    
if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    x = torch.randn(128, 64)
    x = x.to(device)
    classifier = GetClassifier(ClassifierNames.longformer, layers=[92, 128, 128, 128, 128], num_classes=3, device=device)
    pred = classifier(x)
    print(pred.shape)
    '''kan_model = KANClassifier(layers=[64, 128, 128, 128, 128], num_classes=3)
    print(kan_model)
    pred = kan_model(x)
    print(pred.shape)

    mamba_model = MambaClassifier(layers=[64, 128, 128, 128, 128], num_classes=3)   
    print(mamba_model)
    pred = mamba_model(x)
    print(pred.shape)

    tsf_model = TransformerClassifier(layers=[64, 128, 256, 256, 128], num_classes=3)   
    print(tsf_model)
    pred = tsf_model(x)
    print(pred.shape)'''

