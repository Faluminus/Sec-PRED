import tensorflow as tf
from transformers import AutoTokenizer
import numpy as np


class Inference():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.conv = tf.keras.models.load_model('./api/prediction_models/convolutionalSecPRED.keras')
        self.lstm = tf.keras.models.load_model('./api/prediction_models/lstmSecPRED-PREDICTION.keras')
    
    def _predict_lstm(self, ac):
        tokenized_input = self.tokenizer(ac, return_tensors="tf")["input_ids"]
        predictions = self.lstm.predict(tokenized_input)
        predicted_labels = np.argmax(predictions, axis=-1)
        return self._process_output(predicted_labels)
    
    def _predict_conv(self, ac):
        tokenized_input = self.tokenizer(ac, return_tensors="tf")["input_ids"]
        predictions = self.conv.predict(tokenized_input)
        predicted_labels = np.argmax(predictions, axis=-1)
        return self._process_output(predicted_labels)
    
    def _process_output(self, predicted_labels):
        output = self.tokenizer.decode(predicted_labels[0])
        filtered_vals = ['<','>','c','l','s','e','o','s',' ']
        cleared_output = ""
        for x in output:
            if x not in filtered_vals: 
                cleared_output += x
        return cleared_output

    def predict(self, ac, model_type="lstm"):
        if model_type == "lstm":
            return self._predict_lstm(ac)
        else:
            return self._predict_conv(ac)