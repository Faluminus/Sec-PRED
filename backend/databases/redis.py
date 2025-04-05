import redis
import time

class RedisHandler():
    def __init__(self):
        keep_running = True
        while keep_running:
            try:
                pool = redis.ConnectionPool(host='redis', port=6379, db=0, max_connections=10)
                self.redis_db = redis.Redis(decode_responses=True, connection_pool=pool)
                keep_running = False
            except:
                time.sleep(5)
        return self.redis_db
    

class RedisSingleton(RedisHandler):
    __private_instance = None
    __is_initialized = False
    def __new__(cls):
        if cls.__private_instance is None:
            cls.__private_instance = super().__init__(cls)
        return cls.__private_instance
        
    def __init__(self):
        print(self.__private_instance)
        if RedisSingleton.__is_initialized:
            return
        RedisSingleton.__is_initialized = True
        return self.__private_instance
    
    



    




    
