import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from laserOptimizer import *


class InverseSolver:
    """Compute the Lambda

    Attributes:
        _pst (ProblemSettings): Contains grid, physical values etc infomation

    Methods:
        get_Lambda(): returns the values of lambda
        plot_Lambda(): Show a lambda figure all over the body
    """    
    def __init__(self, settings, tarT, tarTarray, curT):
        self._pst = settings
        self.tarT =tarT ##shd be matrix
        self.curT = curT
        self.tarTarray = tarTarray
        self.adjointField = []
    def gauss_quadrature_1D(self,num_points):
    # 获取 Legendre 多项式的零点和权重
        gauss_points, weights = np.polynomial.legendre.leggauss(num_points)
        
        return gauss_points, weights

    def get_Lambda(self):
        # Storage
        Rhs_all = np.zeros(self._pst.nnod) #[nnod]=[51x51]
        I       = np.zeros((self._pst.nel,self._pst.nnodel*self._pst.nnodel)) #[nel,nnodel^2]=[2500,16]
        J       = np.zeros((self._pst.nel,self._pst.nnodel*self._pst.nnodel)) #[nel,nnodel^2]=[2500,16]
        K       = np.zeros((self._pst.nel,self._pst.nnodel*self._pst.nnodel)) #[nel,nnodel^2]=[2500,16]
        GCOORD = self._pst.get_gCoord()
        EL2NOD = self._pst.get_EL2NOD()
        meltpool_T = 1300.0
        
        # Formulation: [K] WITHOUT Mixed Boundary Condition Term + [F]
        for iel in range(0,self._pst.nel):
            ECOORD = np.take(GCOORD, EL2NOD[iel,:], axis=0 ) # [4x2]
            Ael    = np.zeros((self._pst.nnodel,self._pst.nnodel)) #[4x4]
            Rhs_el = np.zeros(self._pst.nnodel)
            
            for ip in range(0,self._pst.nip):        
                # 1. update shape functions
                xi      = gauss[ip,0]
                eta     = gauss[ip,1]
                N, dNds = shapes_2D(xi, eta)
                
                # 2. set up Jacobian, inverse of Jacobian, and determinant
                Jac     = np.matmul(dNds,ECOORD) #[2x2]<-[2,nnodel]*[nnodel,2]=[2,4]x[4,2]
                invJ    = np.linalg.inv(Jac)     
                detJ    = np.linalg.det(Jac)
                
                # 3. get global derivatives
                dNdx    = np.matmul(invJ, dNds) # [2x4]<-[2,2]*[2,nnodel]=[2,2]x[2,4]

                
                # 4. compute element stiffness matrix
                for i in range(0,4):
                    for j in range(0,4):
                        Ael[i][j] += ((self._pst.alpha[0] * N[i] *dNdx[0][j]) - self._pst.beta * (dNdx[0][i] * dNdx[0][j] + dNdx[1][i] * dNdx[1][j])) * detJ             
                # 5. assemble right-hand side 
                Nj = np.zeros((2,4))
                Nj[0,:]=N

                [x,y] = localmaptoglobal_2D(xi,eta, iel)
                tarTfct =int_factory_2D(self.tarT)
                curTfct = int_factory_2D(self.curT)
                localcurT = curTfct(x,y)
                localtarT = tarTfct(x,y)
                
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
                    
                if localtarT >= meltpool_T-10.0 and localtarT <= meltpool_T+100:
                    localtarT = meltpool_T
                    c_manual = 100.0
                '''
###Case 1
                if localtarT >= meltpool_T:
                    localtarT = meltpool_T
                    c_manual = 1.0 
                '''
                T_diff = c_manual*(localcurT - localtarT) ## this is C, manually chnge it acording to Td
    
                Rhs_el[0] += T_diff * N[0] * detJ
                Rhs_el[1] += T_diff * N[1] * detJ
                Rhs_el[2] += T_diff * N[2] * detJ
                Rhs_el[3] += T_diff * N[3] * detJ
                
            Rhs_all[EL2NOD[iel,:]] += Rhs_el

        #for iel in range(0,self._pst.nel):            
            I[iel,:]  =  (EL2NOD[iel,:]*np.ones((self._pst.nnodel,1), dtype=int)).T.reshape(self._pst.nnodel*self._pst.nnodel) #reshape->[4x4]
            J[iel,:]  =  (EL2NOD[iel,:]*np.ones((self._pst.nnodel,1), dtype=int)).reshape(self._pst.nnodel*self._pst.nnodel)   #reshape->[4x4]
            K[iel,:]  =  Ael.reshape(self._pst.nnodel*self._pst.nnodel) #reshape->[4x4]


        # Formulation: [K].Mixed Boundary Condition Term      
        elIndexRHS = []
        for i in range(self._pst.ney):
            new_item = self._pst.nex - 1 + i * self._pst.nex 
            elIndexRHS.append(new_item)  # 将新项添加到列表末尾    
        #print("Elements index on RHS:", elIndexRHS)
        for iel in elIndexRHS:  
            ECOORD = np.take(GCOORD, EL2NOD[iel,:], axis=0 ) # [4x2]
            Ael    = np.zeros((self._pst.nnodel,self._pst.nnodel)) #[4x4]
            Rhs_el = np.zeros(self._pst.nnodel)
            for ip in range(0,2):        
                # 1. update shape functions
                GP_lin, GW_lin =self.gauss_quadrature_1D(2)
                xi      = 1
                eta     = GP_lin[ip]
                N, dNds = shapes_2D(xi, eta)
                
                # 2. set up Jacobian, inverse of Jacobian, and determinant
                Jac     = np.matmul(dNds,ECOORD) #[2,nnodel]*[nnodel,2]
                invJ    = np.linalg.inv(Jac)     
                #detJ    = np.linalg.det(Jac)
                detJ    = self._pst.ly/self._pst.ney/2.0
                
                # 3. get global derivatives
                dNdx    = np.matmul(invJ, dNds) # [2,2]*[2,nnodel]

                # 4. compute element stiffness matrix
                Ntemp = N.reshape(N.size,1)
                #Ael = Ael- (self._pst.v/self._pst.k1 * np.matmul(Ntemp, Ntemp.T)*detJ)
                for i in range(0,4):
                    for j in range(0,4):
                        Ael[i][j] -= self._pst.v / self._pst.k1 * self._pst.c * N[i] * N[j] * detJ
                        #Ael[i][j] -= self._pst.v  * self._pst.c/self._pst.k1 * N[i] * N[j] * detJ
                #        pass

        # assemble coefficients
        #for iel in elIndexRHS:
            #I[iel,:]  =  (EL2NOD[iel,:]*np.ones((self._pst.nnodel,1), dtype=int)).T.reshape(self._pst.nnodel*self._pst.nnodel) #reshape->[4x4]
            #J[iel,:]  =  (EL2NOD[iel,:]*np.ones((self._pst.nnodel,1), dtype=int)).reshape(self._pst.nnodel*self._pst.nnodel)   #reshape->[4x4]
            K[iel,:]  +=  Ael.reshape(self._pst.nnodel*self._pst.nnodel) #reshape->[4x4]
            #pass
        A_all = csr_matrix((K.reshape(self._pst.nel*self._pst.nnodel*self._pst.nnodel),(I.reshape(self._pst.nel*self._pst.nnodel*self._pst.nnodel),J.reshape(self._pst.nel*self._pst.nnodel*self._pst.nnodel))),shape=(self._pst.nnod,self._pst.nnod))
        
        # indices and values at top and bottom
        i_left  = np.arange(0,self._pst.nx*self._pst.ny-self._pst.nx+1,self._pst.nx, dtype=int)

        Ind_bc  = i_left
        Val_bc  = np.zeros(i_left.shape) 

        # smart way of boundary conditions that keeps matrix symmetry
        Free    = np.arange(0,self._pst.nnod)
        Free    = np.delete(Free, Ind_bc)
        TMP     = A_all[:,Ind_bc]
        #Rhs_all = self.tarTarray #Pattern should be the same
        Rhs_all = Rhs_all - TMP.dot(Val_bc)
        Rhs_all[Ind_bc] = 0.0 #force boundary
        
        # solve reduced system
        Lambda = np.zeros(self._pst.nnod)
        Lambda[Free] = spsolve(A_all[np.ix_(Free, Free)],Rhs_all[Free])
        Lambda[Ind_bc] = Val_bc
        result =Lambda.reshape((self._pst.ny,self._pst.nx))
        self.adjointField = result
        return Lambda

    def interpolate_lambda(self, x):
        interp = int_factory_2D(self.adjointField)
        
        
        return interp(x,self._pst.ly-10E-12)
        
    

    def plot_Lambda(self):
        plt.figure()
        L = self.get_Lambda()
        GCOORD = self._pst.get_gCoord()
        X = np.reshape(GCOORD[:, 0], (self._pst.ny, self._pst.nx))
        Y = np.reshape(GCOORD[:, 1], (self._pst.ny, self._pst.nx))
        L = L.reshape((self._pst.ny, self._pst.nx))
        cp = plt.contourf(X, Y, L, cmap='rainbow')
        cbar = plt.colorbar(cp)
        cbar.set_label('Lambda')
        plt.tight_layout()
        plt.title('Lambda over the Domain')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        