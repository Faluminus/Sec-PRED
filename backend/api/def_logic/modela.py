from databases.redis import RedisSingleton
import json
import uuid
import base64

class ModelA():
    def __init__(self):
        #Redis
        self.redis_db = RedisSingleton()
        return None  

    def add_empty_cache_record(self,amino_acid):
        uuid_bytes = uuid.uuid4().bytes
        id = base64.urlsafe_b64encode(uuid_bytes).decode("utf-8").rstrip("=")
        self.redis_db.hset(id, mapping={
            "pending":"True",
            "aminoAcids": "empty",
            "secondaryStructureCONV": "empty",
            "secondaryStructureLSTM": "empty",
        })
        self.redis_db.expire(id, 86400)
        
        self.redis_db.set(amino_acid, id)
        self.redis_db.expire(amino_acid, 86401) 
        return id

    def push_to_queue(self,id,aminoAcids):
        data = json.dumps([id,aminoAcids])
        self.redis_db.rpush("queue",data)
            
    def check_cache_record(self,id = None,amino_acids=None):
        if amino_acids is not None:
            id = self.redis_db.get(amino_acids)
            if id is None:
                return False,None
        pending = self.redis_db.hget(id,"pending")
        if pending == None:
            return False,{'PENDING': None,'ERROR': True}
        if pending == "True":
            return False,{"PENDING":True, 'ERROR': False}
        amino_acids = self.redis_db.hget(id,"aminoAcids")
        secondary_structure_conv = self.redis_db.hget(id,"secondaryStructureCONV")
        secondary_structure_lstm = self.redis_db.hget(id,"secondaryStructureLSTM")
        #xy_visualisation = self.redis_db.hget(id,"xyVisualisation")
        #xy_height = self.redis_db.hget(id, "xyHeight")
        #xy_width = self.redis_db.hget(id, "xyWidth")
        if bytes.isinstance():
            

        return True,{"PENDING": False, 
        "AC": bytes.decode(amino_acids), 
        "SSCONV": bytes.decode(secondary_structure_conv),
        "SSLSTM": bytes.decode(secondary_structure_lstm),
        "ID": bytes.decode(id), 
        'ERROR': False}
    
