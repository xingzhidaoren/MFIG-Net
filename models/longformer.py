'''
author: Jun Wang 
copyright: Hangzhou City University
times:2025.1.13
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LongformerSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, embed_dim = x.size()
        
        # 线性投影得到Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 滑动窗口局部注意力
        attn_output = self.sliding_chunks_attention(q, k, v, attention_mask)
        
        # 合并多头输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # 最终线性投影
        output = self.out_proj(attn_output)
        
        return output
    
    def sliding_chunks_attention(self, q, k, v, attention_mask=None):
        batch_size, num_heads, seq_len, head_dim = q.size()
        
        # 填充序列长度到窗口大小的倍数
        padding_len = (self.window_size - seq_len % self.window_size) % self.window_size
        if padding_len > 0:
            q = F.pad(q, (0, 0, 0, padding_len))
            k = F.pad(k, (0, 0, 0, padding_len))
            v = F.pad(v, (0, 0, 0, padding_len))
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, padding_len), value=0)
        
        padded_seq_len = seq_len + padding_len
        num_windows = padded_seq_len // self.window_size
        
        # 重塑为窗口形式
        q = q.view(batch_size, num_heads, num_windows, self.window_size, head_dim)
        k = k.view(batch_size, num_heads, num_windows, self.window_size, head_dim)
        v = v.view(batch_size, num_heads, num_windows, self.window_size, head_dim)
        
        # 计算每个窗口内的注意力
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, 1, num_windows, self.window_size)
            attention_mask = attention_mask.unsqueeze(1)  # 广播到所有头
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # 重塑回原始形状
        attn_output = attn_output.view(batch_size, num_heads, padded_seq_len, head_dim)
        
        # 移除填充
        if padding_len > 0:
            attn_output = attn_output[:, :, :seq_len, :]
        
        return attn_output

class LongformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, dropout=0.1, feedforward_dim=None):
        super().__init__()
        self.self_attn = LongformerSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size
        )
        
        feedforward_dim = feedforward_dim or 4 * embed_dim  # 默认使用4倍embed_dim
        
        self.linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x, attention_mask=None):
        # 自注意力
        attn_output = self.self_attn(x, attention_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # 前馈网络
        ff_output = self.linear2(self.activation(self.linear1(x)))
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class LongformerEncoder(nn.Module):
    def __init__(self, layer_dims, num_heads=8, window_size=32, max_seq_len=512, dropout=0.1):
        """
        Args:
            layer_dims: 各层的维度列表，如[64, 128, 256]
            num_heads: 注意力头数
            window_size: 局部注意力窗口大小
            max_seq_len: 最大序列长度
            dropout: dropout概率
        """
        super().__init__()
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        
        # 输入投影层
        self.input_proj = nn.Linear(layer_dims[0], layer_dims[0])
        
        # 位置编码
        self.position_embeddings = nn.Embedding(max_seq_len, layer_dims[0])
        
        # 创建编码器层
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = LongformerEncoderLayer(
                embed_dim=layer_dims[i],
                num_heads=num_heads,
                window_size=window_size,
                dropout=dropout
            )
            self.layers.append(layer)
            
            # 如果不是最后一层，添加投影层到下一层的维度
            if i < self.num_layers - 1:
                proj = nn.Linear(layer_dims[i], layer_dims[i+1])
                self.layers.append(proj)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(layer_dims[-1])
    
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()
        
        # 输入投影
        x = self.input_proj(x)
        
        # 添加位置编码
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings
        
        x = self.dropout(x)
        
        # 通过编码器层
        for layer in self.layers:
            if isinstance(layer, LongformerEncoderLayer):
                x = layer(x, attention_mask)
            else:  # 投影层
                x = layer(x)
        
        x = self.norm(x)
        
        return x

  


class LongformerClassifier(nn.Module):
    def __init__(self, layers=[64, 128, 256], num_classes=2,  
                 num_heads=8, window_size=32, max_seq_len=512, dropout=0.2, device='cpu'):
        """
        Args:
            input_dim: 输入特征维度 (len_features)
            num_classes: 类别数
            layer_dims: 各层的维度列表，如[64, 128, 256],第一个是输入层
            num_heads: 注意力头数
            window_size: 局部注意力窗口大小
            max_seq_len: 最大序列长度
            dropout: dropout概率
        """
        super(LongformerClassifier, self).__init__()
        
        # 初始投影层，将输入维度映射到第一层的维度
        self.feature_extractor = torch.nn.Sequential()

        for in_features, out_features in zip(layers, layers[1:]):   
            self.feature_extractor.append(
                nn.Linear(in_features, out_features)
            )
            self.feature_extractor.append(
                nn.ReLU()
            )
       
            self.feature_extractor.append(LongformerEncoder(
                layer_dims=[out_features],
                num_heads=num_heads,
                window_size=window_size,
                max_seq_len=max_seq_len,
                dropout=0.0
            ))       

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
        # x的形状: [batch_size, input_dim]
        x = torch.unsqueeze(x, dim=1)      
        
        # Longformer编码
        x = self.feature_extractor(x)  # [batch_size, seq_len, layer_dims[-1]]
        
        x = x.transpose(1, 2)  # [batch_size, layer_dims[-1], seq_len]
        x = x.squeeze(-1)  # [batch_size, layer_dims[-1]]
        
        # 分类
        logits = self.classfier(x)  # [batch_size, num_classes]
        if return_feature:
            return x, logits
        return logits
    
if __name__ == '__main__':
    # 模型参数
  
    num_classes = 20  # 类别数
    num_features = 123
    layer_dims = [num_features, 128, 128]  # 控制层数和特征数
    batch_size = 16  # 批量大小
    

    # 创建模型
    model = LongformerClassifier(
        layers=layer_dims,
        num_classes=num_classes,        
        num_heads=8,
        window_size=32
    )
    print(model)

    # 创建随机输入数据
    x = torch.randn(batch_size, num_features)

    # 前向传播
    output = model(x)
    print(output.shape)  # 应该输出: torch.Size([16, 10])