from flask import request, jsonify
from api.controller import Controller
from threading import Thread
from flask_restful import Resource
from flasgger import swag_from
import hashlib 

controller = Controller()

class GetById(Resource):
    
    @swag_from({
        'responses': {
            200: {
                'description': 'A status code 200 means successful and returns a list of items.',
                'content': {
                    'application/json': {
                        'examples': {
                            'example1': {
                                'summary': 'Successful response',
                                'output': 'CCCCSSSCCSSCCCCCCCEEEECTTCCEEEEECCTTSCTTTCEEEETTCCCTTHHHHHHHHHHC'
                            }
                        }
                    }
                }
            },
        }
    })
    def get(self,id):
        return controller.get_by_id(id)
