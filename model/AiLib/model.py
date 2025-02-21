class Model:  
    __private_instance = None

    def __new__(cls, *args):
        if cls.__private_instance is None:
            cls.__private_instance = super(Model, cls).__new__(cls,*args)
        return cls.__private_instance
    
    def __init__():
        pass

    def chain():
        pass
