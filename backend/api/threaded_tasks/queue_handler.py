from api.def_logic.model_inference import Inference
#from api.secvis.secvis import SecVis 
from databases.redis import RedisSingleton
import json

class QueueHandler():
    def __init__(self):
        #self.secvis = SecVis()
        #self.secvis.SetDims2D(400,200)
        self.inference = Inference()
        self.redis_db = RedisSingleton()
        self.queue = "queue"
    

    def __call__(self):
        while True:
            query = json.loads(self.__pop_from_queue()[1])
            amino_acids = query[1]
            id = query[0]
            if query is not None:
                secondary_structure_conv = self.inference.predict(ac=amino_acids, model_type="conv")
                secondary_structure_lstm = self.inference.predict(ac=amino_acids, model_type="lstm")
                #xy_visualisation = self.secvis.Draw2D(secondary_structure_conv)
                builder = CacheRecordBuilder()
                builder.amino_acids(amino_acids)
                builder.pending("False")
                builder.secondary_structure_conv(secondary_structure_conv)
                builder.secondary_structure_lstm(secondary_structure_lstm)
                self.__fill_cache_record(builder(), id)


    def __pop_from_queue(self):
        query = self.redis_db.blpop([self.queue],timeout=0)
        return query


    def __fill_cache_record(self, mappings, id):
        self.redis_db.hset(str(id),mapping=mappings)


class CacheRecordBuilder():
    def __init__(self):
        self.mappings = dict()
    
    def __call__(self) -> dict:
        return self.mappings

    def pending(self, pending: str):
        self.mappings["pending"] = pending

    def amino_acids(self, amino_acids):
        self.mappings["aminoAcids"] = amino_acids

    def secondary_structure_conv(self, secondary_structure_conv: str):
        self.mappings["secondaryStructureCONV"] = secondary_structure_conv

    def secondary_structure_lstm(self, secondary_structure_lstm: str):
        self.mappings["secondaryStructureLSTM"] = secondary_structure_lstm

    def clear(self):
        self.mappings = dict()
