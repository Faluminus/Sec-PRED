import numpy as np
import math
import matplotlib as mp 


class SecVis():
    """Returns tensor with coordinates for each structure"""
    def __init__(self,secondaryStructure:str):
        self.secStrucArr = secondaryStructure.split()
        self.drawers = Drawers([400,200])
    def run(self):
        orientations = []
        for x in self.secStrucArr:
            match(x):
                case 'H':
                    orientations.append(self.drawers.AlphaHelix(num_waves=5))
        return orientations

class Drawers():
    def __init__(self,resolution:np.ndarray):
        """resolution - [x pixels,y pixels]"""
        assert resolution[0] > resolution[1] ,"x axis cant be smaller than y axis"
        
        self.x_axis = resolution[0]
        self.y_axis = resolution[1]
 
    def AlphaHelix(self,num_waves:int) ->  tuple[np.ndarray, str]:
        d_y = self.y_axis /2
        y_coordinations = []
        for x_pos in range(self.x_axis):
            y_pos = math.sin(x_pos/(self.x_axis/2*math.pi*num_waves))*d_y
            y_coordinations.append(y_pos)
        return y_coordinations , "red"
    
    def BetaBridge(self) ->  tuple[np.ndarray, str]:
        d_y = self.y_axis /2
        y_coordinations = []
        for x_pos in range(self.x_axis):
            y_pos = 0
            y_coordinations.append(y_pos)
        return y_coordinations,"green"
        

    def BetaLadder(self) ->  tuple[np.ndarray, str]:
        pass

    def GHelix(self) -> tuple[np.ndarray, str]:
        pass

    def PiHelix(self) ->  tuple[np.ndarray, str]:
        pass

    def KHelix(self) -> tuple[np.ndarray, str]:
        pass

    def HydrogenBondedTurn(self) ->  tuple[np.ndarray, str]:
        pass

    def Bend(self) ->  tuple[np.ndarray, str]:
        pass



