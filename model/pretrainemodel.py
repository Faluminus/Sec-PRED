from transformers import BartForConditionalGeneration, AutoTokenizer
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch

# Load tokenizer and device
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
data = pd.read_csv("D:/MyPrograms/Secondary/Sec-PRED/data/raw/ps4dataset.csv")

# Ensure that input and dssp8 columns are lists of strings
src_data = data['input'].astype(str).tolist()
tgt_data = data['dssp8'].astype(str).tolist()

# Tokenize the data
src_data_tokenized = tokenizer(src_data, return_tensors="pt", padding=True, truncation=True, max_length=500)
tgt_data_tokenized = tokenizer(tgt_data, return_tensors="pt", padding=True, truncation=True, max_length=500)

# Create Dataset and DataLoader
train_dataset = TensorDataset(
    src_data_tokenized['input_ids'], 
    src_data_tokenized['attention_mask'], 
    tgt_data_tokenized['input_ids'], 
    tgt_data_tokenized['attention_mask']
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Load model and define optimizer and loss
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=tokenizer.bos_token_id).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.AdamW(model.parameters(), lr=0.001, eps=1e-6, weight_decay=0.001)

# Training loop
model.train()
for epoch in range(20):
    for i,batch in enumerate(train_loader):
        src_train, src_mask, tgt_train, tgt_mask = [x.to(device) for x in batch]
        
        # Forward pass
        outputs = model(
            input_ids=src_train,
            attention_mask=src_mask,
            labels=tgt_train,
            decoder_attention_mask=tgt_mask
        )
        
        # Compute loss and perform backprop
        loss = outputs.loss
        loss.backward()
        if i % 128 == 0:
            optimizer.step()
            optimizer.zero_grad()

        print(f"Batch Loss: {loss.item()}")
    
    print(f"Epoch {epoch + 1} complete, Loss: {loss.item()}")
    torch.save(model.state_dict(), 'bartPROT_epoch_{epoch}.pth')
