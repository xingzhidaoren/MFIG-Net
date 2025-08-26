'''
author: Jun Wang 
copyright: Hangzhou City University
times:2025.2.3
'''
import torch
import torch.nn as nn

    
class TransformerClassifier(nn.Module):
    def __init__(self, layers=[64, 128, 128], num_classes=3, num_heads=4, dropout=0.2, device='cpu'):
        super(TransformerClassifier, self).__init__()
        self.feature_extractor = torch.nn.Sequential()
      
        for in_features, out_features in zip(layers, layers[1:]):   
            self.feature_extractor.append(
                nn.Linear(in_features, out_features)
            )
            self.feature_extractor.append(
                nn.ReLU()
            )

            self.feature_extractor.append(
                nn.TransformerEncoder(nn.TransformerEncoderLayer(out_features, num_heads), 1)                                 
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
    

if __name__ == '__main__':
    # 模型参数
  
    num_classes = 20  # 类别数
    num_features = 121
    layer_dims = [num_features, 128, 128]  # 控制层数和特征数
    batch_size = 16  # 批量大小
    

    # 创建模型
    model = TransformerClassifier(
        layers=layer_dims,
        num_classes=num_classes,        
        num_heads=8
    )
    print(model)

    # 创建随机输入数据
    x = torch.randn(batch_size, num_features)

    # 前向传播
    output = model(x)
    print(output.shape)  # 应该输出: torch.Size([16, 10])