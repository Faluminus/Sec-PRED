from api.model_inference import Inference
from api.secvis.secvis import SecVis 
import redis
import requests
import json

class QueueHandler():
    def __init__(self):
        self.secvis = SecVis()
        self.secvis.SetDims2D(400,200)
        self.inference = Inference()
        self.redis_db = redis.Redis(host='localhost',port=6379,decode_responses=True) 
        self.queue = "queue"
    

    def __call__(self):
        while True:
            query = json.loads(self.__pop_from_queue()[1])
            amino_acids = query[1]
            id = query[0]
            if query is not None:
                secondary_structure_conv = self.inference.predict(ac=amino_acids, model_type="conv")
                secondary_structure_lstm = self.inference.predict(ac=amino_acids, model_type="lstm")
                xy_visualisation = self.secvis.Draw2D(secondary_structure_conv)
                self.__fill_cache_record(id,amino_acids, secondary_structure_conv, secondary_structure_lstm, xy_visualisation, self.secvis.dims2D['x'], self.secvis.dims2D['y'])


    def __pop_from_queue(self):
        query = self.redis_db.blpop([self.queue],timeout=0)
        return query


    def __fill_cache_record(self, id, amino_acids, secondary_structure_conv, secondary_structure_lstm, xy_visualisation, xy_width, xy_height):
        mappings = {
            "pending":"False",
            "aminoAcids":amino_acids,
            "secondaryStructureCONV":secondary_structure_conv,
            "secondaryStructureLSTM":secondary_structure_lstm,
            "xyVisualisation": json.dumps(xy_visualisation),
            "xyWidth": xy_width,
            "xyHeight": xy_height
        }
        self.redis_db.hset(str(id),mapping=mappings)


