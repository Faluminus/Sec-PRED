from api.modela import ModelA
from threading import Thread
import json
from flask import jsonify ,make_response
import json
class Controller:
    def __init__(self):
        self.amino_acids = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        self.model = ModelA()
        self.webhook = "http://localhost:5000/webhook/"
    
    def __data_check(self,data):
        #Check if key AC exists and if dict is empty
        if 'AC' not in data.keys():
            return make_response(jsonify({"output": "Unknown variable provided"}), 400)
        
        #Check if AC contains valid aminoacids
        for i,ac in enumerate(data['AC']):
            if ac.upper() not in self.amino_acids:
                error_output = f'Uknown amino acid type used on char:{i}' 
                return make_response(jsonify({"output": "Unknown amino acid type used on char:{i}"}), 422)
        return None

    def do_prediction(self,data):
        data = json.loads(data.decode('utf-8'))
        status = self.__data_check(data)
        if status is not None:
            return status
        
        aminoAcids = data['AC']
        status,secondaryStructure = self.model.check_cache_record(aminoAcids=aminoAcids)
        if status:
            return make_response(jsonify(secondaryStructure),200)
        
        id = self.model.add_empty_cache_record(aminoAcids)
        self.model.push_to_queue(id,json.dumps(aminoAcids))
        
        return make_response(jsonify({"AC":aminoAcids,"ID":id}),202)
    
    
    def prediction_client_send(self,id,record=None):
        if record is None:
            status,record = self.model.check_cache_record(id=id)
        
        payload = json.dumps(record)
        try:

            response = requests.post(self.webhook+id,json=payload)
        except:
            return
         
    
