from flask_restful import Resource
from flasgger import swag_from


class Welcome(Resource):
    @swag_from({
        'responses': {
            200: {
                'description': 'A status code 200 means successful and returns a message.',
                'content': {
                    'application/json': {
                        'examples': {
                            'example1': {
                                'summary': 'Successful response',
                                'value': {'message': 'Welcome GeeksforGeeks!!'}
                            }
                        }
                    }
                }
            }
        }
    })
    def get(self):
        """
        This is an example endpoint which returns a simple message.
        """
        return {'message': 'SecPRED api'}