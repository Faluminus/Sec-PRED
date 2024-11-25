from api.modela import ModelA
from api.model_inference import Inference
from threading import Thread

from flask import jsonify


class Controller:
    def __init__(self):
        self.amino_acids = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        self.inference = Inference()
        self.model = ModelA()
    
    def __data_check(self,data):
        #Check if key AC exists and if dict is empty
        if 'AC' not in data.keys() and not data:
            return jsonify({"output": "Unknown variable provided"}), 400

        #Check if AC contains valid aminoacids
        for i,ac in enumerate(data['AC']):
            if ac not in self.amino_acids:
                return jsonify({"output": "Unknown amino acid type used on char:{i}"}), 422
        return None

    def create_empty_cache_record(self,data):
        status = self.__data_check(data)
        if status is not None:
            return status;

        self.model.add_empty_cache_record(data['AC'])
        
    def get_structure(self,data):

        status = self.__data_check(data)
        if status is not None:
            return status;
        
        #Check cache for secondary structure
        secstruct = self.model.query_cache(data['AC'])
        if secstruct:
            return jsonify({"output": secstruct}), 200

        #Check db for secondary structure and if finds ---> save to cache
        secstruct = self.model.query_db(data['AC'])
        if secstruct:
            self.model.update_cache(data['AC'],secstruct)
            return jsonify({"output": secstruct}), 200
        
        
        #Predicts secondary structure and updates db with cache
        secstruct = self.inference.predict(data['AC'])
        self.model.update_db(data['AC'],secstruct)
        self.model.update_cache(data['AC'],secstruct)
        
        return jsonify({"output": secstruct}), 200
    