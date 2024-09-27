import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import AutoTokenizer



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=800):
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
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.3,max_len=800):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.tgt_mask = None

        self.pos_encoder = PositionalEncoding(ninp, dropout,max_len)
        self.pos_decoder = PositionalEncoding(ninp, dropout,max_len)

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.tgt_emb = nn.Embedding(ntoken, ninp)

        self.transformer = nn.Transformer(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers, num_decoder_layers=nlayers,batch_first=True)

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
          src_mask = torch.zeros((src.size(1), src.size(1)), dtype=torch.bool).to(src.device)
        if tgt_mask is None:
          tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device).type(torch.bool)




        src_padding_mask = (src == tokenizer.pad_token_id).to(src.device)
        tgt_padding_mask = (tgt == tokenizer.pad_token_id).to(tgt.device)


        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        tgt = self.tgt_emb(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_decoder(tgt)

        memory = self.transformer.encoder(src,src_key_padding_mask=src_padding_mask)
        output = self.transformer.decoder(tgt, memory, memory_mask=src_mask,tgt_key_padding_mask=tgt_padding_mask,memory_key_padding_mask=src_padding_mask)



        output = self.decoder(output)
        return F.log_softmax(output,dim=-1)


def add_token(start_token,output,position,max_seq_len,batch_size):
    """Combines output and start token , used if autoregression is needed"""
    mask = torch.tensor([True if x <= position else False for x in range(max_seq_len)]).unsqueeze(0).repeat(batch_size,tokenizer.cls_token_id)
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

    print(mistakes)
    print(len(hypothesis))
    print(reference_len)
    accuracy = 1 - (mistakes/reference_len)
    return accuracy



#Transformer hyperparams
d_model = 512
d_heads = 8
d_ff = 2048
layers = 2
dropout = 0.3

#Optimizer params
lr = 0.001
weight_decay = 0.0001

#Sheduler params
mode = 'max'
factor = 0.1
patience=5

#Data params
max_seq_length = 500
batch_size = 64


def tokenize_data(data):
        return tokenizer(list(data), return_tensors="pt", padding=True, truncation=True,max_length=500)
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
data = pd.read_csv("/content/drive/MyDrive/AMINtoSEC.csv")
src_data = tokenize_data(data['AminoAcidSeq'])
tgt_data = tokenize_data(data['SecondaryStructureSeq'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



src_data_train, src_data_test, tgt_data_train , tgt_data_test = train_test_split(src_data['input_ids'], tgt_data['input_ids'], test_size=0.20, random_state=42)

split_idx = int(0.8 * len(src_data_train))
trainloader = DataLoader(list(zip(src_data_train[:split_idx], tgt_data_train[:split_idx])), batch_size=batch_size, shuffle=True)
valloader = DataLoader(list(zip(src_data_train[split_idx:], tgt_data_train[split_idx:])), batch_size=batch_size, shuffle=True)




model = TransformerModel(tokenizer.vocab_size,d_model,d_heads,d_ff,layers,dropout,max_seq_length).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-6, weight_decay=weight_decay)
sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)




for epoch in range(200):
  model.train()
  total_loss = 0.
  ntokens = tokenizer.vocab_size
  for i, batch in enumerate(trainloader):
      data, targets = batch[0].to(device), batch[1].to(device)

      tgt_input = torch.full((data.size(0), max_seq_length), tokenizer.cls_token_id, device=device).to(device)


      teacher_forcing_ratio = 0.75
      for t in range(max_seq_length):
          output = model(data, tgt_input)

          if torch.rand(1).item() < teacher_forcing_ratio:
              tgt_input = tgt_input.clone()
              tgt_input[:, t] = targets[:, t]
          else:
              tgt_input = tgt_input.clone()
              tgt_input[:, t] = output[:,t,:].argmax(dim=1)

      loss = criterion(output.view(-1, ntokens), targets.view(-1))
      print(loss)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
      optimizer.step()
      optimizer.zero_grad()
      total_loss += loss.item()

  model.eval()
  scores = []
  with torch.no_grad():
      for i, batch in enumerate(valloader):
          src_data_val, tgt_data_val = batch
          src_data_val = src_data_val.to(device)
          tgt_data_val = tgt_data_val.to(device)

          tgt_input = torch.full((batch.size(0), max_seq_length), tokenizer.cls_token_id, device=device)

          for t in range(max_seq_length):
              output = model(src_data_val, tgt_input)
              tgt_input = tgt_input.clone()
              tgt_input[:, t] = output[:,t,:].argmax(dim=1)
          torch.save(model.state_dict(),'/content/drive/MyDrive/diffmodelComplex.pth')


          for prediction, target in zip(tgt_input, tgt_data_val):
              score = Q8_score(tokenizer.decode(prediction), tokenizer.decode(target))
              scores.append(score)

      percentual_score = sum(scores) / len(scores)

  print(f"Epoch:{epoch} Validation Q8 Score: {percentual_score} Learning rate: {optimizer.param_groups[0]['lr']}")
  sheduler.step(percentual_score)

