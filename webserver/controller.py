# controller.py
class TaskController:
    def __init__(self, model):
        self.model = model
    def predictSecStrucuture(aminoAcid):
        secStructure = self.model.predict(aminoAcid)
        return secStructure
 