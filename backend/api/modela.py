import redis
import json
import uuid
import base64

class ModelA():
    def __init__(self):
        #Redis
        self.redis_db = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.redis_db.set('global_id',-1)
        return None  

    def add_empty_cache_record(self,amino_acid):
        id = self.redis_db.incr('global_id')
        if id == 1:
            self.redis_db.expire('global_id',31536000) #year
        
        self.redis_db.hset(str(id),mapping={
            "pending":"True",
            "aminoAcids": "empty",
            "secondaryStructureCONV": "empty",
            "secondaryStructureLSTM": "empty",
            "xyVisualisation": "empty"
        })
        self.redis_db.expire(str(id),86400) #day
        
        self.redis_db.set(amino_acid,str(id))
        self.redis_db.expire(amino_acid,86401) 
        
        return id


    def push_to_queue(self,id,aminoAcids):
        data = json.dumps([id,aminoAcids])
        self.redis_db.rpush("queue",data)
        
            
    def check_cache_record(self,id=0,amino_acids=None):
        if amino_acids is not None:
            id = self.redis_db.get(amino_acids)
            if id is None:
                return False,None


        pending = self.redis_db.hget(id,"pending")
        amino_acids = self.redis_db.hget(id,"aminoAcids")
        secondary_structure_conv = self.redis_db.hget(id,"secondaryStructureCONV")
        secondary_structure_lstm = self.redis_db.hget(id,"secondaryStructureLSTM")
        xy_visualisation = self.redis_db.hget(id,"xyVisualisation")
        xy_height = self.redis_db.hget(id, "xyHeight")
        xy_width = self.redis_db.hget(id, "xyWidth")

        if pending == None:
            return False,{'ERROR': True}
        if pending == "True":
            return False,{"PENDING":True, 'ERROR': False}
        if pending == "False":
            return True,{"PENDING": False, 
            "AC": amino_acids, 
            "SSCONV": secondary_structure_conv,
            "SSLSTM": secondary_structure_lstm,
            "ID": id, 
            "XY": xy_visualisation,
            "XYHEIGHT": xy_height,
            "XYWIDTH": xy_width,
            'ERROR': False}
        
        return False,None
    
    
