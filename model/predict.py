import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


data = pd.read_csv("./data/raw/data.csv")


tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")


src_data = data['input'].astype(str).tolist()
tgt_data = data['dssp8'].astype(str).tolist()

src_data_tokenized = tokenizer(src_data, padding=True, truncation=True, max_length=500, return_tensors="tf")
tgt_data_tokenized = tokenizer(tgt_data, padding=True, truncation=True, max_length=500, return_tensors="tf")


src_input_ids = src_data_tokenized['input_ids']
tgt_input_ids = tgt_data_tokenized['input_ids']


src_input_ids = tf.where(src_input_ids == 1, 0, src_input_ids)
tgt_input_ids = tf.where(tgt_input_ids == 1, 0, tgt_input_ids)

src_input_ids_np = src_input_ids.numpy()
tgt_input_ids_np = tgt_input_ids.numpy()



src_data_train, src_data_test, tgt_data_train, tgt_data_test = train_test_split(src_input_ids_np, tgt_input_ids_np, test_size=0.20, random_state=42)



max_seq_len = 500
max_features = tokenizer.vocab_size
embedding_dim = 16
num_classes = 9  

model = tf.keras.models.load_model('./convolutionalsecpred.h5')

sample_input = src_data_test[0].reshape(1, -1)  

data = model.predict(sample_input)

for sequence in np.argmax(data, axis=2): 
    decoded_sequence = tokenizer.decode(sequence.tolist(), skip_special_tokens=True)
    print(decoded_sequence)
