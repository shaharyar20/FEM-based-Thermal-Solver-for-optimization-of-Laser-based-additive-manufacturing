import numpy as np
from .ProblemSettings import *


parameters = ProblemSettings()

def extractDofsFace(surface):

    if (surface == "bottom"):
        dofs =np.arange(0,parameters.nx, dtype=int)
        
    elif (surface == "top"):
        dofs =np.arange(parameters.nx*parameters.ny-parameters.nx,parameters.nx*parameters.ny, dtype=int )
            
    elif (surface == "left"):
        dofs =np.arange(0,parameters.nx*parameters.ny-parameters.nx+1,parameters.nx, dtype=int )
                
    elif (surface == "right"):
        dofs =np.arange(parameters.nx-1,parameters.nx*parameters.ny,parameters.nx, dtype=int )
    
    return dofs

totalDofs = parameters.nx*parameters.ny
topDofs  = extractDofsFace("top")
botDofs  = extractDofsFace("bottom")         
leftDofs = extractDofsFace("left")
rightDofs= extractDofsFace("right")

def localmaptoglobal_2D(xi, eta, elId):
    '''
    dx, dy, elemtn size along x,y
    

    '''
    dx = parameters.lx/parameters.nex
    dy = parameters.ly/parameters.ney
    nx = (elId%parameters.nex)
    ny = np.floor(elId/parameters.nex)
    
    x0 = (dx*nx) +(dx/2)
    y0 = (dy*ny) + (dy/2)
    
    x = x0 +(xi*(dx/2))
    y = y0 +(eta*(dy/2))
    if (int(elId>=(parameters.nex*parameters.ney))):
        raise Exception(" element id is wrong!")
        
    if(xi>1 or xi<-1 or eta >1 or eta<-1):
        raise Exception("xi/eta coordinate out of bound")
    
    return x,y

def shapes_2D(xi, eta):
    
    #shape functions
    
    N1 = 0.25*(1-xi)*(1-eta)
    N2 = 0.25*(1+xi)*(1-eta)
    N3 = 0.25*(1+xi)*(1+eta)
    N4 = 0.25*(1-xi)*(1+eta)

    N = np.array([N1, N2, N3, N4])
    
    # and their derivatives
    dNds = np.zeros((2,4))
 
    dNds[0,0]   =  0.25*(-1 + eta) #derivative with xi
    dNds[1,0]   =  0.25*(xi - 1) #derivative with eta

    #derivatives of second shape function with local coordinates
    dNds[0,1]   =  0.25*(1 - eta)
    dNds[1,1]   =  0.25*(-xi - 1)

    #derivatives of third shape function with local coordinates
    dNds[0,2]   =  0.25*(eta  +  1)
    dNds[1,2]   =  0.25*(xi  +  1)

    #derivatives of fourth shape function with local coordinates
    dNds[0,3]   =  0.25*(-eta -  1)
    dNds[1,3]   =  0.25*(1   - xi)
    
    return N, dNds

def shapes_1D(xi):
    
    #shape functions
    
    N1 = 0.5*(1-xi)
    N2 = 0.5*(1+xi)

    N = np.array([N1, N2])
    
    # derivatives
    dNds = np.zeros(2)
 
    dNds[0] = -0.5 #N1derivative with xi
    dNds[1] = 0.5  #N2derivative with xi
    
    #print()

    return N, dNds


def int_factory_2D (u_hat):
    """
    u_hat       :vector of coefficient to interpolate
    """
    def interpolaton_2D (x,y):
        """
        x,y        :sample points (global coordinates)
        u_hat: 0,0 is define at top left, but the element is define at bottom left
        """
        
        dx =parameters.lx/parameters.nex
        dy =parameters.ly/parameters.ney
        
        x_coor = int(np.floor(x/dx)) #no. node
        y_coor = int(np.floor(y/dy))
        
        
        delX = x-(np.floor(x/dx)*dx) #remain x coordate within the element
        delY = y-(np.floor(y/dy)*dy)
        
        
        #if (y_coor==parameters.ney):  ##change the element index if the point is in the domain bounday
        #    y_coor -=1
        #    delY = dy
        
        #if (x_coor==parameters.nex):
        #    x_coor -=1
        #    delX = dx
            
       
        elId = x_coor + y_coor*parameters.nex #get current element ID   

        xi = delX*(2.0/dx)-1
        eta = delY*(2.0/dy)-1
        
        N1 = 0.25*(1-xi)*(1-eta)
        N2 = 0.25*(1+xi)*(1-eta)
        N3 = 0.25*(1+xi)*(1+eta)
        N4 = 0.25*(1-xi)*(1+eta)
        N = np.transpose(np.array([N1,N2,N3,N4]))
        
        #node_1 = x_coor,parameters.ny-1-y_coor        #node coordinate in domain, to get the related u_hat value
        #node_2 = x_coor+1,parameters.ny-1-y_coor
        #node_3 = (x_coor+1),(parameters.ny-1-y_coor-1)
        #node_4 = (x_coor),(parameters.ny-1-y_coor-1)
        
        node_1 = y_coor, x_coor        #node coordinate in domain, to get the related u_hat value
        node_2 = y_coor, x_coor+1
        node_3 = (y_coor+1),(x_coor+1)
        node_4 = (y_coor+1),(x_coor)
        u_hat_temp =np.array([u_hat[node_1],u_hat[node_2],u_hat[node_3],u_hat[node_4]])
        result = N@u_hat_temp
        return result

    return interpolaton_2D

T_hat = np.zeros((parameters.nx,parameters.ny))
T_interpolate =int_factory_2D(T_hat)
Td_2D       = np.ones((parameters.nx,parameters.ny))
Td_interpolate =int_factory_2D(Td_2D)

