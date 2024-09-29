import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

max_features = 100
embedding_dim = 16
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_features, embedding_dim, mask_zero=True),
    tf.keras.layers.Conv1D(32, 19, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),  # Add pooling to flatten the output
    tf.keras.layers.Dense(44, activation='relu'),
    tf.keras.layers.Dense(9, activation='softmax'),  # Softmax for multi-class classification
])

