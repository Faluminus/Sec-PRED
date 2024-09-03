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
                        pos_encoded_matrix[token_loc,2*i,seq_pos] = math.sin(token_loc/math.pow(self.scalar,2*i/self.dim))
                        pos_encoded_matrix[token_loc,2*i+1,seq_pos] = math.cos(token_loc/math.pow(self.scalar,2*i/self.dim))
        
        return pos_encoded_matrix

class  MultiheadSelfAttention(nn.Model):
    def __init__(self,Q,K,V,num_heads,dropout=0.0,bias=True,add_bias_kv=False,add_zero_attn=False,kdim=None,vdim=None,batch_first=False,device=None,dtype=None):
        super().__init__(MultiheadSelfAttention,self)
        self.q_linear = nn.Linear(Q,Q)
        self.k_linear = nn.Linear(K,K)
        self.v_linear = nn.Linear(V,V)

    def forward(self,return_attention=False,mask=None):
        dot_product = self.q_linear.matmul(self.k_linear.T)
        scaled_dot_product = 1.0 / math.sqrt(self.d_model) * dot_product

        if mask is not None:
            scaled_dot_product = scaled_dot_product.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scaled_dot_product,dim=0)
        results = attention.matmul(self.v_linear)

        if return_attention == True:
            return results,attention
        return results
        
        

class EncoderLayer(nn.Model):
    def __init__(self,positional_embedding,multihead_attention):
        super().__init__(EncoderLayer)
        self.linear1 = nn.Linear(nn.LayerNorm(torch.add(positional_embedding,multihead_attention)))
        self.feed_forward = self.forward()
        self.normalization = nn.LayerNorm(torch.add(self.linear1,self.feed_forward))
        
    def forward(self):
        relu = nn.ReLU(self.linear1)
        linear2 = nn.Linear(relu)
        return linear2



class DecoderLayer(nn.Model):
    def __init__(self,output_embedding,encoder_k_q,positional_encoding):
        super().__init__(DecoderLayer)
        self.positional_embedding = torch.add(output_embedding,positional_encoding)
        self.self_attn = MultiheadSelfAttention(K=self.positional_embedding,Q=self.positional_embedding,V=self.positional_embedding)
        self.cross_attn = MultiheadSelfAttention(K=encoder_k_q,Q=encoder_k_q,V=)

    def forward(self,mask):
        self_attn = self.self_attn.start(return_attention=True,mask=mask)
        norm1 = nn.LayerNorm(torch.add(self_attn),self.positional_embedding)
        cross_attn = self.cross_attn()
        


class Transformer(nn.Model):
    def __init__(self):
        super().__init__(Transformer)
    
    def generate_mask(self,tgt):
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return tgt_mask
    


    

