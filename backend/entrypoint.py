from flask import Flask
from api.def_logic.controller import Controller
from flask_restful import Api
from flasgger import Swagger
from api.threaded_tasks.queue_handler import QueueHandler
from flask_cors import CORS
import threading
app = Flask(__name__)
api = Api(app)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
controller = Controller()
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SWAGGER'] = {
    'title': 'My API',
    'uiversion': 3
}

swagger = Swagger(app)

########################################################################

from api.resources.all_structures import GetAllProteins
from api.resources.doprediction import DoPrediction
from api.resources.welcome import Welcome
from api.resources.get_by_id import GetById
from api.resources.fetchPDB import FetchPDB
from api.threaded_tasks.queue_handler import QueueHandler 


GetAllProteins
GetById
DoPrediction
Welcome
FetchPDB

api.add_resource(Welcome, '/api')
api.add_resource(DoPrediction, '/api/do-prediction')
api.add_resource(GetById, '/api/get-by-id/<id>')
api.add_resource(GetAllProteins, '/api/get-all-cached-proteins')
api.add_resource(FetchPDB, '/api/fetch-pdb/<pdb_id>')


########################################################################

if __name__ == '__main__':
    queue_handler = QueueHandler()
    queue_thread = threading.Thread(target=queue_handler,args=())
    queue_thread.start()
    app.run(host='0.0.0.0', port=3000,debug=False) # <----- for production debug FALSE !!!!!!!!!!!
    
