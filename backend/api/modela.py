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
        return None

    def query_cache(self,aminoAcid):
        self.redis_db.hgetall(aminoAcid)
        return None
        
    def query_db(self,aminoAcid):
        self.collection_ac.find_one({"ac":aminoAcid})
        return None

    def update_db(self,aminoAcid,secStruct,drawing2D,drawing3D):
        self.collection_ac.insert_one({"ac":aminoAcid,"sec":secStruct,"drawing2D":drawing2D,"drawing3D":drawing3D})
        return None
        
    def update_cache(self,aminoAcid,secStruct,drawing2D,drawing3D):
        self.redis_db.hset(aminoAcid,mapping={
            'sec':secStruct,
            'drawing2D':drawing2D,
            'drawing3D':drawing3D
        })
        return None