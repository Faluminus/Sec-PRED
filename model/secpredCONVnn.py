import numpy as np
import pandas as pd
import tensorflow as tf
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

#src_data_train_reshaped = np.reshape(src_data_train, (-1, max_seq_len, embedding_dim, 1))
#tgt_data_train_reshaped = np.reshape(tgt_data_train, (-1, max_seq_len, embedding_dim, 1))
#src_data_test_reshaped = np.reshape(src_data_test, (-1, max_seq_len, embedding_dim, 1))
#tgt_data_test_reshaped = np.reshape(tgt_data_test, (-1, max_seq_len, embedding_dim, 1))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_features, output_dim=embedding_dim, mask_zero=True),
    tf.keras.layers.Conv1D(32, 19, activation='relu', padding='same'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tokenizer.vocab_size, activation='softmax')), 
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()
model.fit(src_data_train, tgt_data_train, epochs=10)
model.evaluate(src_data_test, tgt_data_test, batch_size=32)
model.save('convolutionalSecPRED.keras')


