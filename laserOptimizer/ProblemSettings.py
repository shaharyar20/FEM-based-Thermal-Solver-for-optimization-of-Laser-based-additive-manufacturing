import numpy as np

class ProblemSettings:
    """Store Grid, Physical Coefficients Infomation

    Attributes:
        lx (int): x Boudary Length
        ly (int): y Boudary Length

    Methods:

    """
    def __init__(self):
        #Geometry
        self.lx          = 0.1
        self.ly          = 0.03
        self.nx          = 71  #71
        self.ny          = 31   #41
        self.nnodel      = 4
        self.nip         = 4
        self.dx          = self.lx/(self.nx-1)
        self.dy          = self.ly/(self.ny-1)
        self.w_i         = 0.2 # width inclusion
        self.h_i         = 0.2 # heigths inclusion
        self.nex         = self.nx-1
        self.ney         = self.ny-1
        self.nnod        = self.nx*self.ny
        self.nel         = self.nex*self.ney
        self.n_start_top = (self.nx)*(self.ny-1)+1
        
        self.neLaser    = 30
        self.lengthLaser = self.lx
        self.laserCenter = self.lx/2
        
        # model parameters

        self.k1          = 0.136
        self.k2          = 1 #0.001
        self.beta        = self.k1
        self.Ttop        = 1000
        self.Tbot        = 0
        self.v           = 100
        self.c           = 0.472*7.15
        self.alpha       = np.array([self.v * self.c, 0])
        self.T0          = np.ones([self.nnod,1]) #Initial guess


    def get_gCoord(self):       
        GCOORD = np.zeros((self.nnod,2))
        id = 0
        for i in range(0,self.ny):
            for j in range(0,self.nx):
                GCOORD[id,0] = -self.lx/2 + j*self.dx
                GCOORD[id,1] = -self.ly/2 + i*self.dy
                id = id + 1
        return GCOORD

    def get_EL2NOD(self):
        EL2NOD   = np.zeros((self.nel,self.nnodel), dtype=int)
        for iel in range(0,self.nel):
            row        = iel//self.nex   
            ind        = iel + row
            EL2NOD[iel,:] = [ind, ind+1, ind +self.nx+1, ind+self.nx]
        return EL2NOD

    def c_nonlinear(self,T):
        #c0 = 472
        c0=100
        #dc = 101.02*10**(-3)
        dc = 0.0
        return [c0 + T*dc, dc]
        
    def k_nonlinear(self,T):
        #k0 = 13.6
        dk = 0
        k0=0.01
        #dk = 15.3*10**(-3)
        return [k0 + T*dk, dk]
    
    def setElements(self, nx, ny):
        self.nx          = nx
        self.ny          = ny
        self.dx          = self.lx/(self.nx-1)
        self.dy          = self.ly/(self.ny-1)
        self.nex         = self.nx-1
        self.ney         = self.ny-1
        self.nnod        = self.nx*self.ny
        self.nel         = self.nex*self.ney
        self.n_start_top = (self.nx)*(self.ny-1)+1
        self.T0          = np.ones([self.nnod,1]) #Initial guess

# Gauss integration points
nip   = 4
gauss = np.array([[ -np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3), -np.sqrt(1/3)], [-np.sqrt(1/3), -np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3)]]).T.copy()

