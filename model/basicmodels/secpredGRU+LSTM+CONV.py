import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import L2
from sklearn.utils.class_weight import compute_class_weight


data = pd.read_csv("../../data/raw/data.csv")
test_data = pd.read_csv("../../data/raw/cb513.csv")

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
print(test_data.columns)

src_data_ac = data['input'].astype(str).tolist()
tgt_data = data['dssp8'].astype(str).tolist()

src_data_ac_test = test_data['input'].astype(str).tolist()
tgt_data_test = test_data['dssp8'].astype(str).tolist()

src_data_tokenized = tokenizer(src_data_ac, padding=True, truncation=True, max_length=1000, return_tensors="tf")
tgt_data_tokenized = tokenizer(tgt_data, padding=True, truncation=True, max_length=1000, return_tensors="tf")

src_data_tokenized_test = tokenizer(src_data_ac_test, padding=True, truncation=True, max_length=1000, return_tensors="tf")
tgt_data_tokenized_test = tokenizer(tgt_data_test, padding=True, truncation=True, max_length=1000, return_tensors="tf")

src_input_ids = src_data_tokenized['input_ids']
tgt_input_ids = tgt_data_tokenized['input_ids']
src_input_ids_test = src_data_tokenized_test['input_ids']
tgt_input_ids_test = tgt_data_tokenized_test['input_ids']

src_input_ids = tf.where(src_input_ids == 1, 0, src_input_ids)
tgt_input_ids = tf.where(tgt_input_ids == 1, 0, tgt_input_ids)
src_input_ids_test = tf.where(src_input_ids_test == 1, 0, src_input_ids_test)
tgt_input_ids_test = tf.where(tgt_input_ids_test == 1, 0, tgt_input_ids_test)


src_input_ids_np = src_input_ids.numpy()
tgt_input_ids_np = tgt_input_ids.numpy()
src_input_ids_np_test = src_input_ids_test.numpy()
tgt_input_ids_np_test = tgt_input_ids_test.numpy()


classes = np.unique(tgt_input_ids_np.flatten())
weights = compute_class_weight(class_weight='balanced', classes=classes, y=tgt_input_ids_np.flatten())
weights = {cls: weight for cls, weight in zip(classes, weights)}
#class_weights = [[weights[e] for e in row] for row in tgt_input_ids_np]
sample_weights = np.vectorize(weights.get)(tgt_input_ids_np)


src_data_train, src_data_validate, tgt_data_train, tgt_data_validate = train_test_split(
        src_input_ids_np, 
        tgt_input_ids_np,
        test_size=0.20, 
        random_state=42)


classes = np.unique(tgt_data_train.flatten())
weights = compute_class_weight(class_weight='balanced', classes=classes, y=tgt_data_train.flatten())
weights = {cls: weight for cls, weight in zip(classes, weights)}
#class_weights = [[weights[e] for e in row] for row in tgt_input_ids_np]
sample_weights = np.vectorize(weights.get)(tgt_data_train)


max_seq_len = 1000
max_features = tokenizer.vocab_size
embedding_dim = 128
num_classes = 9  



model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_features, output_dim=embedding_dim, mask_zero=True),
    tf.keras.layers.LSTM(128, return_sequences=True, activation='relu', dropout=0.4, kernel_regularizer=L2(0.01)),
    tf.keras.layers.GRU(128, return_sequences=True, activation='relu', dropout=0.4, kernel_regularizer=L2(0.01)),
    tf.keras.layers.Conv1D(64, 70, activation='relu', padding='same' , kernel_regularizer=L2(0.01)),
])


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001 ,clipnorm=0.5)
model.compile(optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            )

history = model.fit(src_data_train, tgt_data_train, validation_data=(src_data_validate,tgt_data_validate), epochs=10, sample_weight=sample_weights)
model.save('./../trained/secpredGRU+LSTM+CONV.keras')
for s,t in zip(src_input_ids_np_test, tgt_input_ids_np_test):
    model.evaluate(s, t)


