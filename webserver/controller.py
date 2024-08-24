# controller.py

class TaskController:
    def __init__(self, model):
        self.model = model
    def predict_sec_strucuture(self,aminoAcid):
        secStructure = self.model.predict(aminoAcid)
        return secStructure
    def get_server_load(self):
        serverLoad = self.model.get_server_load
        return serverLoad
 