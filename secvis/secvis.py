import numpy as np
import math



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
        self.xy_coordinations

    def __str__(self):
        return self.xy_coordinations

    def AlphaHelix(self,num_waves:int) ->  tuple[np.ndarray, str]:
        """Represented via sinus wave \n
        num_waves: integer (number of waves inside addded resolution)
        """
        d_y = self.y_axis /2
        xy_coordinations = []
        for x_pos in range(self.x_axis):
            y_pos = math.sin(x_pos/(self.x_axis/2*math.pi*num_waves))*d_y
            xy_coordinations.append([x_pos,y_pos])
        return xy_coordinations , "red"
    
    def BetaBridge(self,arrow_height:int,arrow_width:int) ->  tuple[np.ndarray, str]:
        """Represented as arow shape \n
        arrow_height: height of triangular shape on end of line \n
        arrow_width: width of triangular shape on end of line
        """
        assert arrow_height <= self.y_axis , "Arrow height cant be bigger than y_axis"
        assert arrow_width <= self.x_axis , "Arrow widht cant be bigger than x_axis "

        d_y = self.y_axis /2
        xy_coordinations = []
        for x_pos in range(self.x_axis - arrow_width):
            xy_coordinations.append([x_pos,d_y])
        for y_pos in range(self.x_axis - arrow_width,self.x_axis):
            for x_pos in range(arrow_height/2,self.y_axis - arrow_height/2,-1):
                xy_coordinations.append([x_pos,y_pos])
        return xy_coordinations,"green"
        
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
    def Thicken(self,thicken_by:int) -> np.ndarray:
        pass



