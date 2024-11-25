from flask_restful import Resource
from flasgger import swag_from


class GetAllProteins(Resource):
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
    def get(self):
        return self.__get_all_proteins()
    def __get_all_proteins():
        pass
    