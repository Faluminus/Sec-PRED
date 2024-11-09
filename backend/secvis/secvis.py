import numpy as np
import math



class SecVis():
    """Returns tensor with coordinates for each structure"""
    def __init__(self,secondaryStructure:str):
        self.secStrucArr = secondaryStructure.split()
        self.drawers2D = None
        self.dims2D = None

        self.alpha_helix_waves = 5
        self.beta_bridge_dotted_lines_spacing = 20
        self.beta_bridge_dotted_line_gaps = 40
        self.beta_bridge_dot_len = 20
        self.beta_ladder_arrow_height = 60
        self.beta_ladder_arrow_width = 60
        #G(3-10) helix is suppossed to have more waves than alpha helix
        self.g_helix_waves = 10
        self.pi_helix_waves = 3

    def SetDims2D(self,x,y):
        self.dims2D = [x,y]
        self.drawers2D = Drawers2D([x,y])
    
    def SetDims3D(self,x,y,z):
        pass
        
    def Draw2D(self):
        assert self.drawers2D !=  None , "First set dimensions with SetDims2D and than visualise :D"

        orientations = []
        for x in self.secStrucArr:
            match(x):
                case 'H':
                    orientations.append(self.drawers2D.AlphaHelix(self.alpha_helix_waves))
                case 'B':
                    orientations.append(self.drawers2D.BetaBridge(self.beta_bridge_dotted_lines_spacing,self.beta_bridge_dotted_line_gaps,self.beta_bridge_dot_len))
                case 'E':
                    orientations.append(self.drawers2D.BetaLadder(self.beta_ladder_arrow_height,self.beta_ladder_arrow_width))
                case 'G':
                    orientations.append(self.drawers2D.GHelix(self.g_helix_waves))
                case 'I':
                    orientations.append(self.drawers2D.PiHelix(self.pi_helix_waves))
                   
        return orientations


class Drawers2D():
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

    def BetaBridge(self,dotted_lines_spacing:int,dotted_line_gaps:int,dot_len:int) ->  tuple[np.ndarray, str]:
        """Represented as two lines connected with dotted lines \n
        that are symboling hydrogen bonds
        """
        assert dotted_line_gaps <= self.x_axis or dotted_line_gaps is not 0, "Gaps in dotted line cannot be bigger than y_axis or 0"
        assert dotted_lines_spacing <= self.x_axis or dotted_lines_spacing is not 0, "Spacing of dotted lines cannot be bigger than x_axis or 0"

        d_y = self.y_axis /2
        xy_coordinations = []
        for x_pos in range(self.x_axis):
            y_pos = math.sin(x_pos/(self.x_axis/math.pi))*d_y
            xy_coordinations.append([x_pos,y_pos])
            xy_coordinations.append([x_pos,-y_pos])
            if x_pos % dotted_lines_spacing == 0:
                for dot_pos_y in range(-y_pos,y_pos):
                    if dot_pos_y % dotted_line_gaps == 0:
                        xy_coordinations.append([x_pos,dot_pos_y])

        return xy_coordinations , "blue"
        
    def BetaLadder(self,arrow_height:int,arrow_width:int) ->  tuple[np.ndarray, str]:
        """Represented as arow shape \n
        arrow_height: height of triangular shape on end of line \n
        arrow_width: width of triangular shape on end of line
        """
        assert arrow_height <= self.y_axis , "Arrow height cant be bigger than y_axis"
        assert arrow_width <= self.x_axis , "Arrow widht cant be bigger than x_axis"
        
        d_y = self.y_axis /2
        xy_coordinations = []
        for x_pos in range(self.x_axis - arrow_width):
            xy_coordinations.append([x_pos,d_y])
        for y_pos in range(self.x_axis - arrow_width,self.x_axis):
            for x_pos in range(arrow_height/2,self.y_axis - arrow_height/2,-1):
                xy_coordinations.append([x_pos,y_pos])
        return xy_coordinations,"green"
        

    def GHelix(self,num_waves:int) -> tuple[np.ndarray, str]:
        """Represented via sinus wave \n
        num_waves: integer (number of waves inside addded resolution)
        """
        d_y = self.y_axis /2
        xy_coordinations = []
        for x_pos in range(self.x_axis):
            y_pos = math.sin(x_pos/(self.x_axis/2*math.pi*num_waves))*d_y
            xy_coordinations.append([x_pos,y_pos])
        return xy_coordinations , "yellow"
        
        
    def PiHelix(self,num_waves) ->  tuple[np.ndarray, str]:
        """Represented via sinus wave \n
        num_waves: integer (number of waves inside addded resolution)
        """
        d_y = self.y_axis /2
        xy_coordinations = []
        for x_pos in range(self.x_axis):
            y_pos = math.sin(x_pos/(self.x_axis/2*math.pi*num_waves))*d_y
            xy_coordinations.append([x_pos,y_pos])
        return xy_coordinations , "pink"
        

    def KHelix(self) -> tuple[np.ndarray, str]:
        pass

    def HydrogenBondedTurn(self) ->  tuple[np.ndarray, str]:
        pass

    def Bend(self) ->  tuple[np.ndarray, str]:
        pass

    def Thicken(self,thicken_by:int) -> np.ndarray:
        pass


class Drawers3D():
    def __init__(self,dimensions:np.ndarray):
        
        pass
    pass


if __name__ == "__main__":
    secvis = SecVis()
    secvis.SetDims2D(400,200)
    output = secvis.Draw2D()