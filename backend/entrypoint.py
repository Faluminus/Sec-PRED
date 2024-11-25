from flask import Flask
from api.controller import Controller
from flask_restful import Api
from flasgger import Swagger

app = Flask(__name__)
api = Api(app)
controller = Controller()

app.config['SWAGGER'] = {
    'title': 'My API',
    'uiversion': 3
}

swagger = Swagger(app)

########################################################################

from api.resources.all_structures import GetAllProteins
from api.resources.structure import GetStructure
from api.resources.welcome import Welcome

GetAllProteins
GetStructure
Welcome

api.add_resource(Welcome, '/')
api.add_resource(GetStructure, '/get-structure')
api.add_resource(GetAllProteins, '/get-all-cached-proteins')

########################################################################

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True) # <----- for production debug FALSE !!!!!!!!!!!

