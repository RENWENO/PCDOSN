#! -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GlobalAttentionPooling


class DotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        
        attention_weights = torch.matmul(q, k.transpose(-2, -1))
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        output = torch.matmul(attention_weights, v)
        return output


class LayerNormDotProductAttentionFusion(nn.Module):
    def __init__(self, input_dim,hidden_size, num_heads):
        super(LayerNormDotProductAttentionFusion, self).__init__()
        
        self.attention_layer = DotProductAttention()
        
        self.layer_norm = nn.LayerNorm((input_dim,), elementwise_affine=True)
        self.liner3 = nn.Linear(hidden_size * num_heads[-1],768)
        self.liner = nn.Linear(768, 1, bias=True)
        self.pool = GlobalAttentionPooling(self.liner)

    def forward(self, x, y,g):
       
        y=self.liner3(y)
        y = self.attention_layer(y, x, x)
        
        fusedx = x+y
        fusedxy = self.layer_norm(fusedx)
        inputs = self.pool(g,fusedxy)
        return inputs

