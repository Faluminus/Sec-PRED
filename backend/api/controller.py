from api.modela import ModelA
from api.model_inference import Inference


from flask import jsonify


class Controller:
    def __init__(self):
        self.amino_acids = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        self.inference = Inference()
        self.model = ModelA()
    
    def get_structure(self,data):

        #Check if key AC exists and if dict is empty
        if 'AC' not in data.keys() and not data:
            return jsonify({"output": "Unknown variable provided"}), 400

        #Check if AC contains valid aminoacids
        for i,ac in enumerate(data['AC']):
            if ac not in self.amino_acids:
                return jsonify({"output": "Unknown amino acid type used on char:{i}"}), 422
        
        #Check cache for secondary structure
        secstruct = self.model.query_cache(data['AC'])
        if secstruct:
            return jsonify({"output": secstruct}), 200

        #Check db for secondary structure and if finds ---> save to cache
        secstruct = self.model.query_cache(data['AC'])
        if secstruct:
            self.model.update_cache(data['AC'],secstruct)
            return jsonify({"output": secstruct}), 200
        
        
        #Predicts secondary structure and updates db with cache
        secstruct = self.inference.predict(data['AC'])
        self.model.update_db(data['AC'],secstruct)
        self.model.update_cache(data['AC'],secstruct)

        return jsonify({"output": secstruct}), 200
    
    