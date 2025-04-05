import psycopg2
import psycopg2.pool
import os
from dotenv import load_dotenv

class PostgreHandler():
    def __init__(self):
        load_dotenv()
        pool = psycopg2.pool.SimpleConnectionPool(2,3, user=os.getenv("POSTGRE_USER"),
                                                   password=os.getenv("POSTGRE_PASSWD"), 
                                                   host='localhost',
                                                   port='5432',
                                                   database='ProteinBD'
                                                    )
        pool = redis.ConnectionPool(host='localhost', port=6379, db=0, max_connections=10)
        self.redis_db = redis.Redis(decode_responses=True, connection_pool=pool)
        return self.redis_db
    

class PostgreSingleton(PostgreHandler):
    __private_instance = None
    __is_initialized = False
    def __new__(cls):
        if cls.__private_instance is None:
            cls.__private_instance = super().__init__(cls)
        return cls.__private_instance

    def __init__(self):
        print(self.__private_instance)
        if PostgreSingleton.__is_initialized:
            return
        PostgreSingleton.__is_initialized = True
        return self.__private_instance
    

singleton = PostgreSingleton()