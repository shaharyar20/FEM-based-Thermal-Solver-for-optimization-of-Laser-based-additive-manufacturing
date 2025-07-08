import numpy as np
import matplotlib.pyplot as plt
from .ProblemSettings import *

parameters = ProblemSettings()

def laserPolynomial (params):
    """
    params       :vector of coefficient to interpolate
    """
    def piecewise_curve (X):
        """
        i_x        :sample points
        """
        result = 0.0
        X0 = parameters.laserCenter-(parameters.lengthLaser/2.0)
        
        result = 0.0
        if (X<X0 or X>(parameters.laserCenter+(0.5*parameters.lengthLaser))):
            result = 0.0
            
        else:
            XLocal = X-X0
            dx = parameters.lengthLaser / parameters.neLaser
            elId =np.floor(XLocal/dx) #get current element
            delX = XLocal - (elId * dx)
            eta= ((delX-(dx/2.0))/(dx/2.0))
            N1=(1-eta)/2.0
            N2=(1+eta)/2.0
            
            globalId = np.array([elId, elId+1]).astype(int)
            result = (N1*params[globalId[0]]) + (N2*params[globalId[1]])
        return result

    return piecewise_curve

def laserGaussian (params):
    """
    n         :number of gauss shape  
    x_domain  :horizontal domain of laser
    u_hat[i]          = P         :Power (height) 
    u_hat[i+n]        = sigma     :half-width 
    u_hat[i+2*n]      = r         :array of center coordinates 
    """        
    #result = 0
    def gaussianCurve (i_x):
        """
        i_x       :Sample point
        """
        result = 0
        n = len(params)//3
        for i in range(0, n):
            P = params[i*3]
            sigma = params[i*3+1]
            r = params[i*3+2]
            result = result + P/(sigma * np.sqrt(2 * np.pi)**2) * (np.exp((-((i_x)-r)**2)/(2 * sigma**2)))          
        return result
    
    
    return gaussianCurve


def projectPiecewise(gaussFn):
    
    coeff = np.zeros((parameters.neLaser+1))
    elLength = parameters.lengthLaser/parameters.neLaser
    el =parameters.neLaser
    matrix = np.zeros((el+1,el+1))
    force = np.zeros((el+1))
    force = force.reshape(force.size,1)
    for i in range(parameters.neLaser):
        elForce = np.zeros((2))
        elMatrix = np.zeros((2,2))
        xi =[-1.0/np.sqrt(3.0), +1.0/np.sqrt(3.0)]
        for j in range(2):

            N1 = 0.5*(1-xi[j])
            N2 = 0.5*(1+xi[j])
            
            N = np.array([N1, N2])
            N = N.reshape(np.size(N),1)
            
            x = (i*elLength)+((elLength*0.5)+(xi[j]*elLength*0.5))
            
            elMatrix += np.matmul(N, N.T)
            elForce[0:2] += N[0:2,0] * gaussFn(x)
            
        matrix[i : i+2, i : i+2] += elMatrix
        force[i :i + 2,0] += elForce[0:2]
        
    coeff = np.linalg.solve(matrix, force)
        
    return coeff

def computeGradient(inverseSolver):
    grad = np.zeros((parameters.neLaser+1))
    
    N1 = lambda xi : (1-xi)/2.0
    N2 = lambda xi : (1+xi)/2.0
    gaussPts = [-1.0/np.sqrt(3.0),1.0/np.sqrt(3.0)] 
    dx = parameters.lengthLaser/parameters.neLaser
    detJ = dx/2.0
    for i in range(0, parameters.neLaser):
        
        for j in range(2):
            N=np.array([N1(gaussPts[j]), N2(gaussPts[j])])
            xLaser = (dx*i) + (gaussPts[j]*dx*0.5) + (dx*0.5)
            xGlobal = (parameters.laserCenter)-(parameters.lengthLaser*0.5) + xLaser
            sensitivity = inverseSolver.interpolate_lambda(xGlobal)
            grad[i:i+2] +=sensitivity*N*detJ
            
    return grad
    


def projectGrad(grad):
    interFn = laserPolynomial(grad)
    return projectPiecewise(interFn)



