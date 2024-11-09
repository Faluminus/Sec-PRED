import tensorflow as tf
from transformers import AutoTokenizer
import numpy as np


class Inference():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.model = tf.keras.models.load_model('./api/prediction_models/convolutionalSecPRED.keras')
        pass
        
    def predict(self,ac):
        tokenized_input = self.tokenizer(ac, return_tensors="tf")["input_ids"]
        predictions = self.model.predict(tokenized_input)
        predicted_labels = np.argmax(predictions, axis=-1)
        output = self.tokenizer.decode(predicted_labels[0])

        #Positional token clearing
        remember = False
        cleared_output = ""
        for x in output:
            if x == '<': 
                remember = False
            if remember:
                cleared_output += x
            if x == '>':
                remember = True
        output = cleared_output.replace(" ", "")
        return output

