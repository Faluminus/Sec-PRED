import tensorflow as tf
import matplotlib as mp
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

data_path:str = "./data/raw/data.csv"

max_tokens:int = 500
batch_size:int = 32
embedding_dim:int = 128
seed:int = 42


tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

data = pd.read_csv(data_path)

src_data = data['input'].astype(str).tolist()
tgt_data = data['dssp8'].astype(str).tolist()

src_data_tokenized = tokenizer(src_data, return_tensors="pt", padding=True, truncation=True, max_length=500)
tgt_data_tokenized = tokenizer(tgt_data, return_tensors="pt", padding=True, truncation=True, max_length=500)

src_input_ids = src_data_tokenized['input_ids']
tgt_input_ids = tgt_data_tokenized['input_ids']

src_input_ids[src_input_ids == 1] = 0
tgt_input_ids[tgt_input_ids == 1] = 0


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(embedding_dim,64,mask_zero=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dense(8192, activation='relu'),
    tf.keras.layers.Dense(8192, activation='relu',),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dense(2048, activation='relu' ),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu' ),
    tf.keras.layers.Dense(512, activation='softmax'),
])


    
