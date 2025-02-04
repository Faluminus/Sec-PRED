import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight


data = pd.read_csv("./../data/raw/data.csv")


tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")


src_data = data['input'].astype(str).tolist()
tgt_data = data['dssp8'].astype(str).tolist()
tgt_lens = np.array([len(v) for v in tgt_data])
print(tgt_lens)

src_data_tokenized = tokenizer(src_data, padding=True, truncation=True, max_length=500, return_tensors="tf")
tgt_data_tokenized = tokenizer(tgt_data, padding=True, truncation=True, max_length=500, return_tensors="tf")

src_input_ids = src_data_tokenized['input_ids']
tgt_input_ids = tgt_data_tokenized['input_ids']


src_input_ids = tf.where(src_input_ids == 1, 0, src_input_ids)
tgt_input_ids = tf.where(tgt_input_ids == 1, 0, tgt_input_ids)

src_input_ids_np = src_input_ids.numpy()
tgt_input_ids_np = tgt_input_ids.numpy()


src_data_train, src_data_test, tgt_data_train, tgt_data_test= train_test_split(
        src_input_ids_np, 
        tgt_input_ids_np,
        test_size=0.20, 
        random_state=42)


max_seq_len = 500
max_features = tokenizer.vocab_size
embedding_dim = 128
num_classes = 9  

#src_data_train_reshaped = np.reshape(src_data_train, (-1, max_seq_len, embedding_dim, 1))
#tgt_data_train_reshaped = np.reshape(tgt_data_train, (-1, max_seq_len, embedding_dim, 1))
#src_data_test_reshaped = np.reshape(src_data_test, (-1, max_seq_len, embedding_dim, 1))
#tgt_data_test_reshaped = np.reshape(tgt_data_test, (-1, max_seq_len, embedding_dim, 1))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_features, output_dim=embedding_dim, mask_zero=True),
    tf.keras.layers.LSTM(128, return_sequences=True, activation='relu',dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.01)),
    tf.keras.layers.Conv1D(32, 19, activation='relu', padding='same'),
])

model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])

model.summary()
history = model.fit(src_data_train, tgt_data_train, validation_data=(src_data_test,tgt_data_test), epochs=15)
model.save('lstmSecPRED-PREDICTION.keras')


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
