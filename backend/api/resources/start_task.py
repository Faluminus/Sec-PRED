from flask import request, jsonify
from api.controller import Controller
from threading import Thread
from flask_restful import Resource
from flasgger import swag_from
import hashlib 

controller = Controller()

class GetStructure(Resource):
    def post(self):
        pass

    def __return_status(self):
        
        

        response = jsonify({'location':f'/prediction-status/{id}'}), 202
        return response