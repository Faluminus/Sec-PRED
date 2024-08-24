from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from urllib.parse import parse_qs
from model import TaskModel
from controller import TaskController


hostName = "localhost"
serverPort = 1212

class HTTPRequests(BaseHTTPRequestHandler):
    def __init__(self,request, client_address, server):
        self.model = TaskModel()
        self.controller = TaskController(self.model)
        super().__init__(request, client_address, server)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        content_type = self.headers.get('Content-Type')

        if content_type == 'application/x-www-form-urlencoded':
            response_data = parse_qs(post_data.decode('utf-8'))
        elif content_type == 'application/json':
            response_data = json.loads(post_data)
            
        
        if 'AC' in response_data and response_data['AC']:
            data = ''.join(response_data['AC'])
            secStructure = self.controller.predict_sec_strucuture(data)
        else:
            secStructure = json.dumps({'error': 'Invalid or missing AC value'})
        
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(bytes(json.dumps({'secStructure': secStructure}), 'utf-8'))

        
            

if __name__ == "__main__":     
    webServer = HTTPServer((hostName, serverPort), HTTPRequests)
    print("Server started http://%s:%s" % (hostName, serverPort))
    
    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
