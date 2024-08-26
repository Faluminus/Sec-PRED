import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math



class PositionalEncoding(nn.Model):
    def __init__(self,d_model,max_seq_len,scalar=10000):
        super().__init__(PositionalEncoding)

        assert d_model.shape()[0] == d_model.shape()[1] , "Dimensions must be same in d_model"
        assert d_model.shape()[0] % 2 == 0 , "Dimension must be divisible by 2"

        self.d_model = d_model
        self.dim = d_model.shape()[0]
        self.max_seq_len = max_seq_len
        self.scalar = scalar

    def encode(self):
        pos_encoded_matrix = torch.zeros(len(seq),self.dim,self.dim)
        for seq_pos,seq in enumerate(self.d_model):
            if len(seq) <= self.max_seq_len:
                for token_loc in range(len(seq)):
                    for i in range(self.dim/2):
                        pos_encoded_matrix[token_loc,2*i+1,seq_pos] = math.sin(token_loc/math.pow(self.scalar,2*i/self.dim))
                        pos_encoded_matrix[token_loc,2*i+1,seq_pos] = math.cos(token_loc/math.pow(self.scalar,2*i/self.dim))
        
        return pos_encoded_matrix

class  MultiheadSelfAttention(nn.Model):
    def __init__(self,d_model,heads):
        super().__init__(MultiheadSelfAttention)
        self.q_linear = nn.Linear(d_model,d_model)
        self.k_linear = nn.Linear(d_model,d_model)
        self.v_linear = nn.Linear(d_model,d_model)

    def scaled_dot_product_attention(self):
        
        dot_product = torch.matmul(self.q_linear,self.k_linear)
        scaled = dot_product / math.sqrt()
    

class Transformer(nn.Model):
    def __init__(self):
        super().__init__(Transformer)
        