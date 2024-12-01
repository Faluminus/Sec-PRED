import pymongo
import redis

class ModelA():
    def __init__(self):

        #MongoDB
        mongoclient = pymongo.MongoClient("mongodb://localhost:27017")
        mydb = mongoclient['SecPRED']
        self.collection_ac = mydb.get_collection('Ac_SecStruct')

        #Redis
        self.redis_db = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.redis_db.set('global_cache_id',0)
        return None
    
    def add_empty_cache_record(self,aminoAcid):
        self.redis_db.incr('global_cache_id',1)
        id = self.redis_db.get('global_cache_id')
        self.redis_db.hset(id,mapping={
            'aminoAcid':aminoAcid,
            'pending':True
            'sec':None,
            'drawing2D':None,
            'drawing3D':None
        })
        self.redis_db.expire(id,86400)
            
    def check_cache_record(self,id):
        max_id = self.redis_db.get('global_cache_id')
        if id < 0 or id > max_id:
            return None

        vals = self.redis_db.hget(id)
        if not vals['pending']:
            return vals
        
        return None