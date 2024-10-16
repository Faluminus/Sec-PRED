from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from transformers import AutoTokenizer
import numpy as np

app = Flask(__name__)


amino_acids = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", 
               "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]


tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = tf.keras.models.load_model('./models/convolutionalsecpred.keras')

@app.route("/predict-structure", methods=["POST"])
def predict_structure():
    data = request.get_json()
    
    for ac in data['AC']:
        if ac not in amino_acids:
            return jsonify({"output": "Unknown amino acid type used"}), 400

    try:
        tokenized_input = tokenizer(data['AC'], return_tensors="tf")["input_ids"]
        predictions = model.predict(tokenized_input)
        
        predicted_labels = np.argmax(predictions, axis=-1)
        output = tokenizer.decode(predicted_labels[0])
        return jsonify({"output": output})
    except:
        return jsonify({"output": "none"})

if __name__ == '__main__':
    app.run(debug=True)
