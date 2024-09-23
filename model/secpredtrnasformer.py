import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")



class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.tgt_mask = None

        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.pos_decoder = PositionalEncoding(ninp, dropout)  

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.tgt_emb = nn.Embedding(ntoken, ninp)  

        self.transformer = nn.Transformer(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers, num_decoder_layers=nlayers)

        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.uniform_(self.tgt_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        
        if src_mask is None:
            src_mask = torch.zeros((src.size(0), src.size(0))).to(src.device).type(torch.bool)
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)

        

        src_padding_mask = (src == tokenizer.pad_token_id).T.to(src.device).float()
        tgt_padding_mask = (tgt == tokenizer.pad_token_id).T.to(tgt.device).float()

        src_padding_mask = src_padding_mask * (-1e9) + (1-src_padding_mask)
        tgt_padding_mask = tgt_padding_mask * (-1e9) + (1-tgt_padding_mask)

        
        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        tgt = self.tgt_emb(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_decoder(tgt)

        memory = self.transformer.encoder(src, src_mask,src_key_padding_mask=src_padding_mask)
        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=src_mask,tgt_key_padding_mask=tgt_padding_mask,memory_key_padding_mask=src_padding_mask)

        output = self.decoder(output) 
        return F.log_softmax(output,dim=-1)


def add_token(start_token,output,position):
    """Combines output and start token , used if autoregression is needed"""
    mask = torch.tensor([True if x <= position else False for x in range(500)]).unsqueeze(0).repeat(32,1)
    indices = mask.nonzero(as_tuple=True)
    start_token[indices] = output[indices]
    return start_token

def batch_decode(output):
    """Converts batch of output values into tokens"""
    decoded = []
    for seq in output:
        decoded.append(torch.argmax(seq,dim=1).unsqueeze(0))
    return torch.cat(decoded, dim=0)
    

def Q8_score(hypothesis,references):
    """Simple accuracy mesure , percentual similarity to reference"""
    mistakes = 0
    references =''.join(references[5:]).replace(' ', '')
    references = references[:references.index('<eos>')]
    reference_len = len(references)
    if '<eos>' in hypothesis:
        hypothesis = hypothesis[:hypothesis.index('<eos>')].replace(' ','')
    else:
        hypothesis = hypothesis.replace(' ','')

    for i,(x,y) in enumerate(zip(references,hypothesis)):
      if x != y:
        mistakes += 1
    accuracy = 1 - (mistakes/reference_len)
    return accuracy

def tokenize_data(data):
        return tokenizer(list(data), return_tensors="pt", padding=True, truncation=True,max_length=500)



#Transformer hyperparams
d_model = 512
d_heads = 4
d_ff = 2048
layers = 2
dropout = 0.3

#Optimizer params
lr = 0.001
weight_decay = 1e-4

#Sheduler params
mode = 'max'
factor = 0.1
patience=5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_csv("C:/Users/semra/Documents/MyPrograms/Sec-PRED/data/raw/data.csv")
src_data = tokenize_data(data['input'])
tgt_data = tokenize_data(data['dssp8'])


src_data_train, src_data_test, tgt_data_train , tgt_data_test = train_test_split(src_data['input_ids'], tgt_data['input_ids'], test_size=0.20, random_state=42)

split_idx = int(0.8 * len(src_data_train))
trainloader = DataLoader(list(zip(src_data_train[:split_idx], tgt_data_train[:split_idx])), batch_size=32, shuffle=True)
valloader = DataLoader(list(zip(src_data_train[split_idx:], tgt_data_train[split_idx:])), batch_size=32, shuffle=True)




model = TransformerModel(tokenizer.vocab_size,d_model,d_heads,d_ff,layers,dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-06, weight_decay=weight_decay)
sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)

print(tokenizer.pad_token_id)

for epoch in range(200):
  model.train()
  total_loss = 0.
  ntokens = tokenizer.vocab_size
  for i,batch in enumerate(trainloader):
      data, targets = batch[0].to(device),batch[1].to(device)
      optimizer.zero_grad()
      output = model(data,targets)
      output = output.view(-1,ntokens)

      loss = criterion(output, targets.view(-1)))

      loss.backward()
      print(loss)
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
      optimizer.step()
      total_loss += loss.item()
  model.eval()
  scores = []
  with torch.no_grad():
    for i, batch in enumerate(valloader):
            src_data_val, tgt_data_val = batch
            src_data_val = src_data_val.to(device)
            tgt_data_val = tgt_data_val.to(device)
            
            seq = torch.cat((torch.tensor([0]).unsqueeze(0).repeat(32,1),torch.tensor([1]).unsqueeze(0).repeat(32,499)),dim=1).to(device)
            for x in range(500):
                output = model(src_data_val,seq)
                tokenized_batch = batch_decode(output)
                seq = add_token(seq,tokenized_batch,x)
                
            for prediction, target in zip(seq, tgt_data_val):
                score = Q8_score(tokenizer.decode(torch.argmax(prediction,dim=1), dim=1),tokenizer.decode(target))
                scores.append(score)
    
    percentual_score = sum(scores) / len(scores)
    
  print(f"Epoch:{epoch} Validation Q8 Score: {percentual_score} Learning rate: {optimizer.param_groups[0]['lr']}")
  sheduler.step(percentual_score)
  torch.save(model.state_dict(),'diffmodelComplex.pth')
