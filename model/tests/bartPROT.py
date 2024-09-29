from transformers import BartForConditionalGeneration, AutoTokenizer
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch

data = 'PLPSPPSKTSLDIAEELQNDKGVSFAFQAREEELGAFTKRTLFAYSGDGLTGPFKAPASAELSSFLTAHPKGRWLIAFPLGTGIVSVDEGILTLEISRSLPEVGSGSSFYLTEK'


tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
src_data = tokenizer(data, return_tensors="pt", padding=True, truncation=True, max_length=500)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0).to(device)
model.load_state_dict(torch.load('./bartPROT.pth',weights_only=True))
model.eval()

input_ids = src_data["input_ids"].to(device)
attention_mask = src_data["attention_mask"].to(device)
with torch.no_grad():
    output = model(input_ids=input_ids, attention_mask=attention_mask)




print(torch.argmax(output.logits,dim=2))

