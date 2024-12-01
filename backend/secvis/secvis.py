import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


class SecVis():
    """Returns tensor with coordinates for each structure"""
    def __init__(self):
        # inits
        self.secStrucArr = None
        self.drawers2D = None
        self.drawers3D = None
        self.dims2D = None
        self.dims3D = None

        #Function parameters
        self.alpha_helix_waves = 5
        self.beta_bridge_dotted_lines_spacing = 40
        self.beta_bridge_dotted_line_gaps = 12
        self.beta_bridge_dot_len = 20
        self.beta_ladder_arrow_height = 70
        self.beta_ladder_arrow_width = 140
        #G(3-10) helix is suppossed to have more waves than alpha helix
        self.g_helix_waves = 7
        self.pi_helix_waves = 3
        

        #Returnables
        self.end = 0
        self.orientations = []



    def SetDims2D(self,x,y):
        self.dims2D = {'x':x,'y':y}
        self.drawers2D = Drawers2D(np.array([x,y]))
    
    def SetDims3D(self,x,y,z):
        self.dims3D = {'x':x,'y':y,'z':z}
        self.drawers3D = Drawers3D(np.array([x,y,z]))
        pass
        
    
    def Draw3D(self,secondaryStructure:str):
        assert self.drawers3D !=  None , "First set dimensions with SetDims2D and than visualise :D"

        self.end = 0
        self.secStrucArr = list(secondaryStructure)
        
        for x in self.secStrucArr:
            match(x):
                case 'H':
                    output = self.drawers3D.AlphaHelix()

            for x in output:
                self.orientations.append(x)
                #Adds z dimensions to end var so next protein starts on the end of the last
                self.end += self.dims3D['z']
                self.drawers3D.SetEnd(self.end)

        return self.orientations


    def Draw2D(self,secondaryStructure:str):

        assert self.drawers2D !=  None , "First set dimensions with SetDims2D and than visualise :D"

        self.secStrucArr = list(secondaryStructure)
        
        
        for x in self.secStrucArr:
            match(x):
                case 'H':
                    output = self.drawers2D.AlphaHelix(self.alpha_helix_waves)
                case 'B':
                    output = self.drawers2D.BetaBridge(self.beta_bridge_dotted_lines_spacing,self.beta_bridge_dotted_line_gaps,self.beta_bridge_dot_len)
                case 'E':
                    output = self.drawers2D.BetaLadder(self.beta_ladder_arrow_height,self.beta_ladder_arrow_width)
                case 'G':
                    output = self.drawers2D.GHelix(self.g_helix_waves)
                case 'I':
                    output = self.drawers2D.PiHelix(self.pi_helix_waves)
                case 'T':
                    output = self.drawers2D.HydrogenBondedTurn()
                case 'S':
                    output = self.drawers2D.Bend()
                case 'C':
                    output = self.drawers2D.KHelix()
            for x in output:
                self.orientations.append(x)
            #Adds x dimensions to end var so next protein starts on the end of the last
            self.end += self.dims2D['x']
            self.drawers2D.SetEnd(self.end)

        return self.orientations


class Drawers2D():
    def __init__(self,resolution:np.ndarray):
        """resolution - [x pixels,y pixels]"""
        assert resolution[0] > resolution[1] ,"x axis cant be smaller than y axis"
        
        self.x_axis = resolution[0]
        self.y_axis = resolution[1]
        self.end = 0

    def SetEnd(self,end:int):
        self.end = end

    def AlphaHelix(self,num_waves:int) -> tuple[np.ndarray]:
        """Represented via sinus wave \n
        num_waves: integer (number of waves inside addded resolution)
        """
        d_y = self.y_axis /2
        xy_coordinations = []
        for x_pos in range(self.end * 10, (self.end + self.x_axis) * 10):
            x_pos = x_pos/10
            y_pos = np.sin(x_pos / (self.x_axis / (2 * math.pi * num_waves))) * d_y
            if x_pos == 20:
                print(y_pos)
            xy_coordinations.append([x_pos,y_pos])
        return xy_coordinations

    def BetaBridge(self,dotted_lines_spacing:int,dotted_line_gaps:int,dot_len:int) ->   tuple[np.ndarray]:
        """Represented as two lines connected with dotted lines \n
        that are symboling hydrogen bonds
        """
        assert dotted_line_gaps <= self.x_axis or dotted_line_gaps == 0, "Gaps in dotted line cannot be bigger than y_axis or 0"
        assert dotted_lines_spacing <= self.x_axis or dotted_lines_spacing == 0, "Spacing of dotted lines cannot be bigger than x_axis or 0"

        d_y = self.y_axis /2
        xy_coordinations = []
        for x_pos in range(self.end,self.end + self.x_axis):
            y_pos = np.sin(x_pos / (self.x_axis / math.pi)) * d_y
            xy_coordinations.append([x_pos,y_pos])
            xy_coordinations.append([x_pos,-y_pos])
            if x_pos % dotted_lines_spacing == 0:
                for dot_pos_y in range(int(-y_pos),int(y_pos)):
                    if dot_pos_y % dotted_line_gaps == 0:
                        xy_coordinations.append([x_pos,dot_pos_y])

        return xy_coordinations
        
    def BetaLadder(self,arrow_height:int,arrow_width:int) ->  tuple[np.ndarray]:
        """Represented as arow shape \n
        arrow_height: height of triangular shape on end of line \n
        arrow_width: width of triangular shape on end of line
        """
        assert arrow_height <= self.y_axis , "Arrow height cant be bigger than y_axis"
        assert arrow_width <= self.x_axis , "Arrow widht cant be bigger than x_axis"
        
        d_y = 0
        xy_coordinations = []
        for x_pos in range(self.end , self.end + self.x_axis - arrow_width):
            xy_coordinations.append([x_pos,d_y])
        
        const = (arrow_height/2)/arrow_width
        for x_pos in range(self.end + self.x_axis - arrow_width,self.end + self.x_axis):
            triangulify = int((x_pos - self.end -self.x_axis + arrow_width)*const)
            for y_pos in range(int(d_y - arrow_height/2) + triangulify,int(d_y+arrow_height/2) - triangulify):
                xy_coordinations.append([x_pos,y_pos])

        return xy_coordinations
        

    def GHelix(self,num_waves:int) ->  tuple[np.ndarray]:
        """Represented via sinus wave \n
        num_waves: integer (number of waves inside addded resolution)
        """
        d_y = self.y_axis /2
        xy_coordinations = []
        for x_pos in range(self.end * 10 , (self.end + self.x_axis)*10):
            x_pos = x_pos/10
            y_pos = np.sin(x_pos / (self.x_axis / (2 * math.pi * num_waves))) * d_y
            xy_coordinations.append([x_pos,y_pos])
        return xy_coordinations

        
        
    def PiHelix(self,num_waves) -> tuple[np.ndarray]:
        """Represented via sinus wave \n
        num_waves: integer (number of waves inside addded resolution)
        """
        d_y = self.y_axis /2
        xy_coordinations = []
        for x_pos in range(self.end * 10, (self.end + self.x_axis)*10):
            x_pos = x_pos/10
            y_pos = np.sin(x_pos / (self.x_axis / (2 * math.pi * num_waves))) * d_y
            xy_coordinations.append([x_pos,y_pos])
        return xy_coordinations
        
    def KHelix(self) ->  tuple[np.ndarray]:
        """Represented via sinus wave \n
        num_waves: integer (number of waves inside addded resolution)
        """
        d_y = self.y_axis /2
        xy_coordinations = []
        for x_pos in range(self.end * 10, (self.end + self.x_axis)*10):
            x_pos = x_pos/10
            y_pos = np.sin(x_pos / (self.x_axis / (2 * math.pi * 5))) * np.sin(x_pos / (self.x_axis / (2 * math.pi * 2))) * d_y
            xy_coordinations.append([x_pos,y_pos])
        return xy_coordinations

    def HydrogenBondedTurn(self) ->   tuple[np.ndarray]:
        """Represented as arow shape \n
        arrow_height: height of triangular shape on end of line \n
        arrow_width: width of triangular shape on end of line
        """
        d_y = 0
        xy_coordinations = []
        for x_pos in range(self.end , self.end + self.x_axis):
            xy_coordinations.append([x_pos,d_y])

        return xy_coordinations

    def Bend(self) ->  tuple[np.ndarray, str]:
        d_y = 0
        xy_coordinations = []
        for x_pos in range(self.end , self.end + self.x_axis):
            xy_coordinations.append([x_pos,d_y])
        return xy_coordinations
        

    def Thicken(self,thicken_by:int) -> np.ndarray:
        pass


class Drawers3D():
    def __init__(self,resolution:np.ndarray):
        self.x_axis = resolution[0]
        self.y_axis = resolution[1]
        self.z_axis = resolution[2]
        self.end = 0
        pass
    
    def SetEnd(self,end:int):
        self.end = end

    def BetaSheet():
        pass

    def AlphaHelix(self):
        xyz_coordinations = []
        z_pos = self.end
        while z_pos <= (self.end+self.z_axis):
            for x_pos in range(self.x_axis,0,-10):
                z_pos+=1
                y_pos = np.sqrt(np.power(self.x_axis,2) - np.power(x_pos,2))
                xyz_coordinations.append([x_pos,y_pos,z_pos])
            for x_pos in range(0,self.x_axis,10):
                z_pos+=1
                y_pos = np.sqrt(np.power(self.x_axis,2) - np.power(-x_pos,2))
                xyz_coordinations.append([-x_pos,y_pos,z_pos])
            for x_pos in range(self.x_axis,0,-10):
                z_pos+=1
                y_pos = np.sqrt(np.power(self.x_axis,2) - np.power(x_pos,2))
                xyz_coordinations.append([-x_pos,-y_pos,z_pos])
            for x_pos in range(0,self.x_axis,10):
                z_pos+=1
                y_pos = np.sqrt(np.power(self.x_axis,2) - np.power(-x_pos,2))
                xyz_coordinations.append([x_pos,-y_pos,z_pos])
            
        return xyz_coordinations
    
    def BetaSheet(self):
        pass

if __name__ == "__main__":
    """
    secvis = SecVis()
    secvis.SetDims2D(600,200)
    output = secvis.Draw2D("HTIBEGSC")
    x, y = zip(*output) 

    plt.plot(x, y, linestyle='None', marker='o', color='blue')  # 'o' is for circular markers

    # Add titles and labels
    plt.title('Scatter Plot of Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.show()
    """
    
    secvis = SecVis()
    secvis.SetDims3D(800,800,700)
    output = secvis.Draw3D("H")
    x, y ,z = zip(*output)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    plt.show()
    
