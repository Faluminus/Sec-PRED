import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight


data = pd.read_csv("../data/raw/data.csv")
test_data = pd.read_csv("../data/raw/cb513.csv")

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
print(test_data.columns)

src_data_ac = data['input'].astype(str).tolist()
tgt_data = data['dssp8'].astype(str).tolist()

src_data_ac_test = test_data['input'].astype(str).tolist()
tgt_data_test = test_data['dssp8'].astype(str).tolist()



src_data_tokenized = tokenizer(src_data_ac, padding=True, truncation=True, max_length=500, return_tensors="tf")
tgt_data_tokenized = tokenizer(tgt_data, padding=True, truncation=True, max_length=500, return_tensors="tf")

src_data_tokenized_test = tokenizer(src_data_ac_test, padding=True, truncation=True, max_length=500, return_tensors="tf")
tgt_data_tokenized_test = tokenizer(tgt_data_test, padding=True, truncation=True, max_length=500, return_tensors="tf")

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


src_data_train, src_data_validate, tgt_data_train, tgt_data_validate = train_test_split(
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
    tf.keras.layers.Conv1D(32, 19, activation='relu', padding='same', input_shape=(None,embedding_dim)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tokenizer.vocab_size, activation='softmax')), 
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
print("fit")
model.summary()
model.fit(src_data_train, tgt_data_train, epochs=5, validation_data=(src_data_validate, tgt_data_validate))
model.evaluate(src_input_ids_np_test, tgt_input_ids_np_test, batch_size=32)
model.save('convolutionalSecPRED-PREDICTION.keras')

predictions = model.predict(src_input_ids_np_test)
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
