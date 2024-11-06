from flask import Flask, request, jsonify
from api.controller import Controller


app = Flask(__name__)
controller = Controller()

@app.route("/get-structure", methods=["POST"])
def get_structure():
    return controller.get_structure(request.data)

@app.route("/get-all-cached-proteins",methods=["GET"])
def get_all_proteins():
    if request.method == 'GET':
       pass 

if __name__ == '__main__':
    app.run(debug=True) # <----- for production debug FALSE !!!!!!!!!!!
