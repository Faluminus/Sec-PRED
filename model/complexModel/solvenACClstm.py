import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.regularizers import L2
from sklearn import preprocessing
import keras


def standardize_seq(seq):
    mean = np.mean(seq)
    std = np.std(seq)
    if std == 0:
        return (seq - mean)
    return (seq - mean) / std

data = pd.read_csv("../data/processed/AMINtoSECwithX_fraction.csv")

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

src_data_ac = data['AminoAcidSeq'].astype(str).tolist()
tgt_data = data['SolventAcessibility'].astype(str).tolist()

src_data_tokenized = tokenizer(src_data_ac, padding=True, truncation=True, max_length=500, return_tensors="tf")

src_data_fin = []
tgt_data_fin = []


for i, (seq_a, seq_b) in enumerate(zip(tgt_data, src_data_tokenized['input_ids'])):
    seq_a = seq_a.split('|')
    processed_seq_a = [int(char.replace(' ', '')) for char in seq_a if char.strip()]
    if len(processed_seq_a) > 0:
        processed_seq_a = processed_seq_a[:500] + [0] * (500 - len(processed_seq_a))
    else:
        continue
    if len(seq_b) > 0:
        processed_seq_b = tf.where(seq_b == 1, 0, seq_b)
    else:
        continue
    tgt_data_fin.append(processed_seq_a)
    src_data_fin.append(processed_seq_b)


src_input_ids_np = np.array(src_data_fin)
tgt_input_ids_np = np.array(tgt_data_fin)


print(src_input_ids_np.shape)
print(tgt_input_ids_np.shape)
src_data_train, src_data_test, tgt_data_train, tgt_data_test = train_test_split(
        src_input_ids_np, 
        tgt_input_ids_np,
        test_size=0.20, 
        random_state=42)


max_seq_len = 500
max_features = tokenizer.vocab_size
embedding_dim = 256
num_classes = 9  

#src_data_train_reshaped = np.reshape(src_data_train, (-1, max_seq_len, embedding_dim, 1))
#tgt_data_train_reshaped = np.reshape(tgt_data_train, (-1, max_seq_len, embedding_dim, 1))
#src_data_test_reshaped = np.reshape(src_data_test, (-1, max_seq_len, embedding_dim, 1))
#tgt_data_test_reshaped = np.reshape(tgt_data_test, (-1, max_seq_len, embedding_dim, 1))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_features, output_dim=embedding_dim, mask_zero=True),
    tf.keras.layers.LSTM(128, return_sequences=True, activation='relu', dropout=0.4, kernel_regularizer=L2(0.01)),
    tf.keras.layers.Conv1D(300, 19, activation='relu', padding='same' , kernel_regularizer=L2(0.01)),
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001 ,clipnorm=0.5)
model.compile(optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.summary()
history = model.fit(src_data_train, tgt_data_train, validation_data=(src_data_test,tgt_data_test), epochs=3)
model.save('lstmSecPREDACC-PREDICTION.keras')


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()
