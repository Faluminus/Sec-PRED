import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Input, Embedding, Conv1D,Conv2D, Dense, Concatenate, TimeDistributed, RepeatVector, Layer, Reshape, LSTM, GlobalAveragePooling2D
from tensorflow.keras.models import Model

data = pd.read_csv("../data/processed/AMINtoSECcleared.csv")

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

src_data_ac = data['AminoAcidSeq'].astype(str).tolist()
src_data_pH = data['PH_normalized'].astype(float).tolist()
#src_data_kelvin = data['Kelvin_standardized'].astype(float).tolist()
tgt_data = data['SecondaryStructureSeq'].astype(str).tolist()

src_data_tokenized = tokenizer(src_data_ac, padding=True, truncation=True, max_length=1000, return_tensors="tf")
tgt_data_tokenized = tokenizer(tgt_data, padding=True, truncation=True, max_length=1000, return_tensors="tf")

src_input_ids = src_data_tokenized['input_ids']
tgt_input_ids = tgt_data_tokenized['input_ids']

src_input_ids = tf.where(src_input_ids == 1, 0, src_input_ids)
tgt_input_ids = tf.where(tgt_input_ids == 1, 0, tgt_input_ids)

src_input_ids_np = src_input_ids.numpy()
src_data_pH_np = np.array(src_data_pH)
#src_data_kelvin_np = np.array(src_data_kelvin)
tgt_input_ids_np = tgt_input_ids.numpy()

src_data_train_ac, src_data_test_ac, src_data_train_pH, src_data_test_pH, tgt_data_train, tgt_data_test = train_test_split(
        src_input_ids_np, 
        src_data_pH_np,
        #src_data_kelvin_np,
        tgt_input_ids_np,
        test_size=0.20, 
        random_state=42)

max_seq_len = 1000
max_features = tokenizer.vocab_size
embedding_dim = 256
num_classes = 9  

#src_data_train_reshaped = np.reshape(src_data_train, (-1, max_seq_len, embedding_dim, 1))
#tgt_data_train_reshaped = np.reshape(tgt_data_train, (-1, max_seq_len, embedding_dim, 1))
#src_data_test_reshaped = np.reshape(src_data_test, (-1, max_seq_len, embedding_dim, 1))
#tgt_data_test_reshaped = np.reshape(tgt_data_test, (-1, max_seq_len, embedding_dim, 1))


class SqueezingLayer(Layer):
    def call(self, x):
        return tf.squeeze(x)

class ArgMaxLayer(Layer):
    def call(self, x):
        print(tf.argmax(x))
        return tf.argmax(x)


input_ac = Input(shape=(max_seq_len,), dtype=tf.int32, name="input_ac")
input_pH = Input(shape=(1,), dtype=tf.float32, name="input_pH")

extended_pH = RepeatVector(max_seq_len)(input_pH)
reshaped_ac = Reshape((max_seq_len, 1))(input_ac)

embedding = Embedding(input_dim=max_features, output_dim=embedding_dim, mask_zero=True)(input_ac)

#lstm1 = LSTM(256, activation='relu')(embedding)
#outputs = LSTM(128, activation='softmax')(lstm1)

conv1 = Conv1D(32, 19, activation='relu', padding='same')(embedding)
#conv2 = Conv1D(64, 9, activation='relu', padding='same')(conv1)

merged = Concatenate()([conv1, extended_pH])
dense1 = Dense(256, activation='relu')(merged)
outputs = TimeDistributed(Dense(tokenizer.vocab_size, activation='softmax'))(dense1)

model = Model(inputs=[input_ac, input_pH], outputs=outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


model.fit([src_data_train_ac,src_data_train_pH], tgt_data_train, epochs=3)
model.evaluate([src_data_train_ac,src_data_train_pH], tgt_data_test, batch_size=32)
model.save('convolutionalSecPRED-PREDICTION.keras')

predictions = model.predict([src_data_train_ac,src_data_train_pH])
predicted_labels = np.argmax(predictions, axis=-1) 


true_labels = tgt_data_test.flatten()
predicted_labels = predicted_labels.flatten()

mask = true_labels != 0
true_labels = true_labels[mask]
predicted_labels = predicted_labels[mask]

cm = confusion_matrix(true_labels, predicted_labels, labels=[i for i in range(num_classes)])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[f'{tokenizer.decode(i)}' for i in range(num_classes)], 
            yticklabels=[f'{tokenizer.decode(i)}' for i in range(num_classes)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Protein Secondary Structure Prediction')
plt.show()
