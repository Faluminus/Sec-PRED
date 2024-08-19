from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import json
from urllib.parse import parse_qs
from model import TaskModel
from controller import TaskController


hostName = "localhost"
serverPort = 8080

class HTTPRequests(BaseHTTPRequestHandler):
    def __init__(self):
        self.model = TaskModel()
        self.controller = TaskController(self.model)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        content_type = self.headers.get('Content-Type')

        if content_type == 'application/x-www-form-urlencoded':
            parsed_data = parse_qs(post_data.decode('utf-8'))
            response_data = self.handle_form_data(parsed_data)
        elif content_type == 'application/json':
            parsed_data = json.loads(post_data)
            response_data = self.handle_json_data(parsed_data)
        
        self.wfile.write(bytes(json.dumps(response_data), "utf-8"))

        #if self.headers['AC'] != '':
            #secStructure = self.controller.predictSecStrucuture(self.headers['AC'])
            

if __name__ == "__main__":     
    webServer = HTTPServer((hostName, serverPort), HTTPRequests)
    print("Server started http://%s:%s" % (hostName, serverPort))
    
    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
