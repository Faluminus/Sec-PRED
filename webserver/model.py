from msa_transformer_load import get_msa_transformer_prediction
import os

# model.py
class TaskModel: 
    def predict(_,aminoAcid):
        secStructure = get_msa_transformer_prediction(aminoAcid)
        return secStructure
    def get_server_load():
        serverLoad = os.getloadavg()
        return serverLoad


    