import torch
import pandas as pd
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
data = tokenizer("RPDFCLEPPYTGPCKARIIRYFANAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA", return_tensors="pt", padding=True, truncation=True,max_length=500)
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esmfold_v1")
model.eval()
output = model(data['input_ids'])
print(torch.argmax(output,dim=1))




