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


def clear_positional_tokens(output:str):
    remember = False
    cleared_output = ""
    for x in output:
        if x == '<': 
            remember = False
        if remember:
            cleared_output += x
        if x == '>':
            remember = True
    return cleared_output

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
        output = clear_positional_tokens(output)
        return jsonify({"output": output})
    except:
        return jsonify({"output": "none"})

@app.route("/get-all-proteins",methods=["GET"])
def get_all_proteins():
    if request.method == 'GET':
        



if __name__ == '__main__':
    app.run(debug=True)
