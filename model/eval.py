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
from tensorflow.keras.metrics import F1Score
import tensorboard
import keras

test_data = pd.read_csv("../data/raw/cb513.csv")



tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

src_data_ac_test = test_data['input'].astype(str).tolist()
tgt_data_test = test_data['dssp8'].astype(str).tolist()

src_data_tokenized_test = tokenizer(src_data_ac_test, padding=True, truncation=True, max_length=1000, return_tensors="tf")
tgt_data_tokenized_test = tokenizer(tgt_data_test, padding=True, truncation=True, max_length=1000, return_tensors="tf")

src_input_ids_test = src_data_tokenized_test['input_ids']
tgt_input_ids_test = tgt_data_tokenized_test['input_ids']

src_input_ids_test = tf.where(src_input_ids_test == 1, 0, src_input_ids_test)
tgt_input_ids_test = tf.where(tgt_input_ids_test == 1, 0, tgt_input_ids_test)

src_input_ids_np_test = src_input_ids_test.numpy()
tgt_input_ids_np_test = tgt_input_ids_test.numpy()

model = tf.keras.models.load_model('./lstmSecPRED-PREDICTION.keras')

for s,t in zip(src_input_ids_test, tgt_input_ids_np_test):
    s = np.reshape(s, (1,1000))
    t = np.reshape(t, (1, 1000))
    model.evaluate(s, t)
