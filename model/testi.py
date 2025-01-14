import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example input sequences
sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

# Pad sequences
padded_sequences = pad_sequences(sequences, padding='post')
print(padded_sequences)