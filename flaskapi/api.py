from flask import Flask,request,jsonify
import tensorflow as tf
from transformers import AutoTokenizer
import numpy as np

app = Flask(__name__)
amino_acids = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", 
               "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = tf.keras.models.load_model('./convolutionalsecpred.h5')


@app.route("/predict-structure",methods=["POST"])
def predict_structure():
    data = request.get_json()
    for ac in data['aminoAcid']:
        if ac not in amino_acids:
            return jsonify("output":"Uknow amino acid type used")
    output = tokenizer.decode(np.argmax(model.predict(tokenizer.tokenize(data['aminiAcid'])),axis=2))
    return jsonify(output)
    
