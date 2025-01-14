import redis
import json

class ModelA():
    def __init__(self):

        #Redis
        self.redis_db = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.redis_db.set('global_id',-1)
        return None  

    def add_empty_cache_record(self,aminoAcid):
        id = self.redis_db.incr('global_id')
        if id == 1:
            self.redis_db.expire('global_id',31536000) #year
        
        self.redis_db.hset(str(id),mapping={
            "pending":"True",
            "aminoAcids": "empty",
            "secondaryStructure": "empty"
        })
        self.redis_db.expire(str(id),86400) #day
        
        self.redis_db.set(aminoAcid,str(id))
        self.redis_db.expire(aminoAcid,86401) 
        
        return id

    def push_to_queue(self,id,aminoAcids):
        data = json.dumps([id,aminoAcids])
        self.redis_db.rpush("queue",data)
        
            
    def check_cache_record(self,id=0,aminoAcids=None):
        if aminoAcids is not None:
            id = self.redis_db.get(aminoAcids)
            if id is None:
                return False,None


        pending = self.redis_db.hget(id,"pending")
        aminoAcids = self.redis_db.hget(id,"aminoAcids")
        secondaryStructure = self.redis_db.hget(id,"secondaryStructure")

        if pending == None:
            return False,None
        if pending == "True":
            return False,{"pending":True}
        if pending == "False":
            return True,{"pending":False,"aminoAcids":aminoAcids,"secondaryStructure":secondaryStructure}
        
        return False,None
    
    
