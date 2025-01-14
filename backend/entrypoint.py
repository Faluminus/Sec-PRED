from flask import Flask
from api.controller import Controller
from flask_restful import Api
from flasgger import Swagger
from api.queue_handler import QueueHandler
import requests
import threading
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
from api.resources.doprediction import DoPrediction
from api.resources.welcome import Welcome
from api.resources.get_webhook import GetWebhook
from api.queue_handler import QueueHandler 

GetWebhook
GetAllProteins
DoPrediction
Welcome

api.add_resource(Welcome, '/api')
api.add_resource(DoPrediction, '/api/do-prediction')
api.add_resource(GetAllProteins, '/api/get-all-cached-proteins')
api.add_resource(GetWebhook,'/webhook/<int:id>')


########################################################################

if __name__ == '__main__':
    queue_handler = QueueHandler()
    queue_thread = threading.Thread(target=queue_handler,args=())
    queue_thread.start()
    app.run(host='0.0.0.0', port=5000,debug=True) # <----- for production debug FALSE !!!!!!!!!!!
    
