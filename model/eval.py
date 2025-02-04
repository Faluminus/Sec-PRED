import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


data = pd.read_csv("../data/raw/data.csv")


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


src_data_train, src_data_test, tgt_data_train, tgt_data_test, tgt_lens_train, tgt_lens_test = train_test_split(
        src_input_ids_np, 
        tgt_input_ids_np,
        tgt_lens,
        test_size=0.20, 
        random_state=42)


max_seq_len = 500
max_features = tokenizer.vocab_size
embedding_dim = 128
num_classes = 9 

model = tf.keras.models.load_model('convolutionalSecPRED-PREDICTION.keras')

predictions = model.predict(src_data_test)
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