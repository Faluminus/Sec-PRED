from api.modela import ModelA
from threading import Thread
import json
from flask import jsonify ,make_response
import json
class Controller:
    def __init__(self):
        self.amino_acids = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        self.model = ModelA()
    
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
        amino_acids = data['AC']
        status, secondary_structure = self.model.check_cache_record(amino_acids=amino_acids)
        if status:
            return make_response(jsonify(secondary_structure),202)

        id = self.model.add_empty_cache_record(amino_acids)
        self.model.push_to_queue(id,json.dumps(amino_acids))
        return make_response(jsonify({"AC": amino_acids, "PENDING": True, "ID": id}),202)
    

    def get_by_id(self,id):
        try:
            id = int(id)
        except:
            return make_response(jsonify({"ERROR": "id is of wrong type"}))

        status, secondary_structure = self.model.check_cache_record(id)
        if status:
            return make_response(jsonify(secondary_structure),200)
        return make_response(jsonify({"PENDING": True, "ERROR": None}))
        
    
