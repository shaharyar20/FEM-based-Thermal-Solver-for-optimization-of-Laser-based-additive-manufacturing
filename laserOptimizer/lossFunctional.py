import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from matplotlib import cm
from .ProblemSettings import *
#from .solvers import *
from .helpers import *

pst = ProblemSettings()

def loss_trapz (T,Td):
    '''
    Perform the numerical integration with np.trapz
    
    Input:      Continous T_hat, Td
    Return:     Scalar value of integration
    '''
    
    X=np.arange(0,pst.lx+pst.dx,pst.dx)
    Y=np.arange(0,pst.ly+pst.dy,pst.dy)
    loss = np.zeros((pst.ny,pst.nx))
    for i in range(pst.ny):
        for j in range(pst.nx): 
            loss[i,j] = 0.5 * (T[i,j] - Td[i,j])**2*10000
    
    return np.sum(np.sum(loss))*pst.dy*pst.dx
    #return np.trapz(np.trapz(loss,Y, axis = 0), X, axis = 0)

def loss_gauss(curT, tarT):
        # Storage
        '''
        Compute loss integration through gaussian integration
        Also added different case, if normal optimization is desired, 
        simply comment the different case, and put c_manual = 1.0
        
        Input:     current Temperature profile(should be from forward solver), target temperature profile
        Return:    Scalar value of integration     
        '''
        loss = 0.0
        I       = np.zeros((pst.nel,pst.nnodel*pst.nnodel)) #[nel,nnodel^2]=[2500,16]
        J       = np.zeros((pst.nel,pst.nnodel*pst.nnodel)) #[nel,nnodel^2]=[2500,16]
        K       = np.zeros((pst.nel,pst.nnodel*pst.nnodel)) #[nel,nnodel^2]=[2500,16]
        GCOORD = pst.get_gCoord()
        EL2NOD = pst.get_EL2NOD()
        meltpool_T = 1300.0
        
        # Formulation: [K] WITHOUT Mixed Boundary Condition Term + [F]
        for iel in range(0,pst.nel):
            ECOORD = np.take(GCOORD, EL2NOD[iel,:], axis=0 ) # [4x2]
            Ael    = np.zeros((pst.nnodel,pst.nnodel)) #[4x4]
            
            for ip in range(0,pst.nip):        
                # 1. update shape functions
                xi      = gauss[ip,0]
                eta     = gauss[ip,1]
                N, dNds = shapes_2D(xi, eta)
                
                Jac     = np.matmul(dNds,ECOORD) #[2x2]<-[2,nnodel]*[nnodel,2]=[2,4]x[4,2]
                invJ    = np.linalg.inv(Jac)     
                detJ    = np.linalg.det(Jac)


                [x,y] = localmaptoglobal_2D(xi,eta, iel)
                tarTfct =int_factory_2D(tarT)
                curTfct = int_factory_2D(curT)
                localtarT = tarTfct(x,y)
                localcurT = curTfct(x,y)
                c_manual = 0.0
###Case2 : Band in meltpool

                if localcurT < 0:
                    localtarT = 0.0
                    c_manual = 100.0
                    
                if localcurT <= meltpool_T and localtarT >= (meltpool_T+100):
                    #inside meltpool error check
                    localtarT = meltpool_T
                    c_manual = 100.0
                    
                if localcurT >= meltpool_T and localtarT <= (meltpool_T-100):
                    #outside meltpool error check
                    localtarT = meltpool_T
                    c_manual = 100.0
                    
                if localtarT >= meltpool_T-100.0 and localtarT <= meltpool_T+100:
                    #main condition
                    localtarT = meltpool_T
                    c_manual = 100.0
                '''
###Case 1:
                if localtarT >= meltpool_T:
                    localtarT = meltpool_T
                    c_manual = 1.0 
                '''    
                T_diff = (localcurT - localtarT)**2
                loss += T_diff * 0.5  * detJ * c_manual
                
        return loss*10000 #is c
