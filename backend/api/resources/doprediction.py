from flask import request, jsonify
from api.controller import Controller
from threading import Thread
from flask_restful import Resource
from flasgger import swag_from
import hashlib 

controller = Controller()

class DoPrediction(Resource):
    
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
            202: {
                'description': 'A status code 202 means accepted and sends link in header.',
                'content': {
                    'application/json': {
                        
                    }
                }
            }
        }
    })
    def post(self):
        return controller.do_prediction(request.data)
