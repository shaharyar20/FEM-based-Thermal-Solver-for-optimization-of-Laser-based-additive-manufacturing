import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from laserOptimizer import *

class ForwardSolver:
    def __init__(self, parms, laser):
        self.nx = parms.nx
        self.ny = parms.ny
        self.lx = parms.lx
        self.ly = parms.ly
        self.dx = parms.dx
        self.dy = parms.dy
        self.T_bot = parms.Tbot
        self.T_top = parms.Ttop
        self.k1 = parms.k1
        self.k2 = parms.k2
        self.w_i = parms.w_i
        self.h_i = parms.h_i
        self.alpha = parms.alpha
        self.beta = parms.beta
        self.nnodel = 4
        self.nip = 4
        self.nnod = parms.nnod
        self.GCOORD = self.get_gCoord()
        self.nel=parms.nel
        self.nex=parms.nex
        self.ney=parms.ney
        self.c=parms.c
        self.v=parms.v
        self.T0=parms.T0
        self.laser = laser
        self.vmax = 5600
        self.vmin = 0
        
    def c_nonlinear(self,T):
        c0 = 472
        dc = 101.02*10**(-3)
        #dc = 0.0
        return [c0 + T*dc, dc]
      
    def k_nonlinear(self,T):
        k0 = 13.6
        #dk = 0
        dk = 15.3*10**(-3)
        return [k0 + T*dk, dk]

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
            EL2NOD[iel,:] = [ind, ind+1, ind+self.nx+1, ind+self.nx]
        return EL2NOD
    
    def get_F(self):
        GCOORD = self.get_gCoord()
        EL2NOD = self.get_EL2NOD()
        nnodel_t = 2
        nnod_t    = self.nx
        nel_t     = self.nex      ###changed
        GCOORD_t  = np.zeros((nnod_t,1)) ###changed
        
        #Global coordinates top surface
        id_t = 0
        for i in range(0,self.nx): ####changed
                GCOORD_t[id_t] = i*self.dx 
                #GCOORD_t[id_t,1] = i*dy
                id_t           = id_t + 1
            
        #Element to nodes connectivity
        EL2NOD_t = np.zeros((nel_t, nnodel_t), dtype=int)
        for iel_t in range(0, nel_t):
            EL2NOD_t[iel_t,:] = [iel_t, iel_t+1]

        #Integration points setup top
        nip_t = 2
        gauss_t = np.array([-np.sqrt(1/3), np.sqrt(1/3)]).T.copy()

        u_hat = np.arange(0,1)
        u_new_hat=np.ones(np.size(u_hat))*50

        flux = laserPolynomial(self.laser)

        F = np.zeros((self.nx*self.ny))
        
        for iel_t in range (0, nel_t):
            ECOORD_t  = np.take(GCOORD_t, EL2NOD_t[iel_t,:], axis=0) #retrieve element coordinates
            
            Fel_t = np.zeros((nnodel_t))
            
            for ip_t in range(0, nip_t):
                
                #use the shape functions
                xi_t      = gauss_t[ip_t]
                N_t, dNds_t = shapes_1D(xi_t)
        
                #setup Jacobian
                Jac_t     = np.matmul(dNds_t, ECOORD_t) #element length/element size in para. space
                invJ_t    = 1/Jac_t
                detJ_t    = Jac_t
            
                #get global derivatives
                
                dNdx_t    = invJ_t*dNds_t
                elid = iel_t + self.nex*(self.ney-1)
                x_global,y_global = localmaptoglobal_2D(xi_t, 1, elid)
            
                Fel_t = Fel_t + N_t * flux(x_global) * detJ_t 
                
            Global_id = EL2NOD[elid,:]   
            Global_top_id = Global_top_id = Global_id[-1:-3:-1]
            F[Global_top_id] += Fel_t 
        return F
        
    def get_T(self):
        '''
        Return: [0]: T in matrix, and [1]T in vector
        '''

        
        GCOORD = self.get_gCoord()
        EL2NOD = self.get_EL2NOD()
        # Storage
        Rhs_all = np.zeros(self.nnod)
        I       = np.zeros((self.nel,self.nnodel*self.nnodel))
        J       = np.zeros((self.nel,self.nnodel*self.nnodel))
        K       = np.zeros((self.nel,self.nnodel*self.nnodel))
#loss =0
        for iel in range(0,self.nel):
            ECOORD = np.take(GCOORD, EL2NOD[iel,:], axis=0 )
            Ael    = np.zeros((self.nnodel,self.nnodel))
            Rhs_el = np.zeros(self.nnodel)
            
            for ip in range(0,nip):        
                # 1. update shape functions
                xi      = gauss[ip,0]
                eta     = gauss[ip,1]
                N, dNds = shapes_2D(xi, eta)
                
                # 2. set up Jacobian, inverse of Jacobian, and determinant
                Jac     = np.matmul(dNds,ECOORD) #[2,nnodel]*[nnodel,2]
                invJ    = np.linalg.inv(Jac)     
                detJ    = np.linalg.det(Jac)
                
                # 3. get global derivatives
                dNdx    = np.matmul(invJ, dNds) # [2,2]*[2,nnodel]

                # 4. compute element stiffness matrix                
                Nj = np.zeros((2,4))
                Nj[0,:]=N
                Ael     = Ael + np.matmul(dNdx.T,dNdx) * detJ * self.k1 + np.matmul(Nj.T,dNdx) * detJ * self.v * self.c
                
                #loss+= （interpolate(T, xi)-interpolate(Ttarget, x))2 detJ
                # 5. assemble right-hand side, no source terms, just here for completeness
                Rhs_el     = Rhs_el + np.zeros(self.nnodel)
            
            # assemble coefficients
            I[iel,:]  =  (EL2NOD[iel,:]*np.ones((self.nnodel,1), dtype=int)).T.reshape(self.nnodel*self.nnodel)
            J[iel,:]  =  (EL2NOD[iel,:]*np.ones((self.nnodel,1), dtype=int)).reshape(self.nnodel*self.nnodel)
            K[iel,:]  =  Ael.reshape(self.nnodel*self.nnodel)
            
            Rhs_all[EL2NOD[iel,:]] += Rhs_el

        A_all = csr_matrix((K.reshape(self.nel*self.nnodel*self.nnodel),(I.reshape(self.nel*self.nnodel*self.nnodel),J.reshape(self.nel*self.nnodel*self.nnodel))),shape=(self.nnod,self.nnod))
        
        F = self.get_F()

        u_dirichlet =np.zeros(totalDofs)
        u_dirichlet[leftDofs]=0.0
        
        u_neumann = np.zeros (totalDofs)
        u_neumann[topDofs]=1.0
        
        forceDirichlet =A_all.dot(u_dirichlet) 
        forceNeumann= np.ones(np.size(u_neumann))*0.0
        forceNeumann[botDofs] = 0.0
        

        finalForce = F - forceDirichlet - forceNeumann
        finalForce[leftDofs] = 0.0
        
        A_all[leftDofs,:]=0.0
        A_all[:,leftDofs]=0.0
        A_all[leftDofs,leftDofs]=1.0

        
        T=spsolve(A_all,finalForce)
        result =T.reshape((self.ny,self.nx))
        return result, T

    def plot_T(self, T):
        X = np.reshape(self.GCOORD[:,0], (self.ny,self.nx))
        Y = np.reshape(self.GCOORD[:,1], (self.ny,self.nx))
        plt.ion()
        fig1 = plt.figure()
        cp1 = plt.contourf(X, Y, T, cmap="turbo")
        plt.colorbar(cp1)
        plt.title('Temperature within the domain')
        plt.xlabel("x")
        plt.ylabel("y")
        aspect_ratio = (self.ly*3) / (self.lx)
        plt.gca().set_aspect(aspect_ratio)
        plt.show()
        plt.ioff()
    
    def get_Residual(self,T):
        GCOORD = self.get_gCoord()
        EL2NOD = self.get_EL2NOD()
        # Storage
        Rhs_all = np.zeros(self.nnod)
        I       = np.zeros((self.nel,self.nnodel*self.nnodel))
        J       = np.zeros((self.nel,self.nnodel*self.nnodel))
        K       = np.zeros((self.nel,self.nnodel*self.nnodel))
        
        #Residual = np.zeros([self.nnod,1])
        Fel = np.zeros([self.nnod,1])
        
        
        for iel in range(0,self.nel):
            ECOORD = np.take(GCOORD, EL2NOD[iel,:], axis=0 )
            Ael    = np.zeros((self.nnodel,self.nnodel))
            Rhs_el = np.zeros(self.nnodel)
            
            for ip in range(0,nip):        
                # 1. update shape functions
                xi      = gauss[ip,0]
                eta     = gauss[ip,1]
                N, dNds = shapes_2D(xi, eta)
                
                # 2. set up Jacobian, inverse of Jacobian, and determinant
                Jac     = np.matmul(dNds,ECOORD) #[2,nnodel]*[nnodel,2]
                invJ    = np.linalg.inv(Jac)     
                detJ    = np.linalg.det(Jac)
                
                # 3. get global derivatives
                dNdx    = np.matmul(invJ, dNds) # [2,2]*[2,nnodel]

                # 4. compute element stiffness matrix                
                Nj = np.zeros((2,4))
                Nj[0,:]=N
                
                [c, cprime] = self.c_nonlinear(np.matmul(N,T[EL2NOD[iel,:]])[0])
                [k, kprime] = self.k_nonlinear(np.matmul(N,T[EL2NOD[iel,:]])[0])
                
                com1 = np.matmul((np.matmul(dNdx, T[EL2NOD[iel,:]])).T,dNdx)*detJ*k
                com2_test = np.matmul(dNdx[0,:], T[EL2NOD[iel,:]])[0]*N*detJ*self.v*c
                
                Rhs_el     = Rhs_el + com1 + com2_test
   
            Rhs_all[EL2NOD[iel,:]] += Rhs_el[0,:]
            
            
        return Rhs_all
        
    def get_T_nonlin(self):
        GCOORD = self.get_gCoord()
        EL2NOD = self.get_EL2NOD()
        # Storage
        Rhs_all = np.zeros(self.nnod)
        I       = np.zeros((self.nel,self.nnodel*self.nnodel))
        J       = np.zeros((self.nel,self.nnodel*self.nnodel))
        K       = np.zeros((self.nel,self.nnodel*self.nnodel))
        T = np.zeros([self.nnod,1])
        Residual = np.zeros([self.nnod,1])
        
        while (1):    
            for iel in range(0,self.nel):
                ECOORD = np.take(GCOORD, EL2NOD[iel,:], axis=0 )
                Ael    = np.zeros((self.nnodel,self.nnodel))
                Rhs_el = np.zeros(self.nnodel)
                
                for ip in range(0,nip):        
                    # 1. update shape functions
                    xi      = gauss[ip,0]
                    eta     = gauss[ip,1]
                    N, dNds = shapes_2D(xi, eta)
                    
                    # 2. set up Jacobian, inverse of Jacobian, and determinant
                    Jac     = np.matmul(dNds,ECOORD) #[2,nnodel]*[nnodel,2]
                    invJ    = np.linalg.inv(Jac)     
                    detJ    = np.linalg.det(Jac)
                    
                    # 3. get global derivatives
                    dNdx    = np.matmul(invJ, dNds) # [2,2]*[2,nnodel]
    
                    # 4. compute element stiffness matrix                
                    Nj = np.zeros((2,4))
                    Nj[0,:]=N
                    
                    [c, cprime] = self.c_nonlinear(np.matmul(N,T[EL2NOD[iel,:]])[0])
                    [k, kprime] = self.k_nonlinear(np.matmul(N,T[EL2NOD[iel,:]])[0])
                    
                    N.resize(np.size(N),1)
                    
                    com1 =  ((np.matmul(dNdx.T,dNdx) *k)+ (np.matmul(N,np.matmul((np.matmul(dNdx, T[EL2NOD[iel,:]])).T, dNdx))*kprime))*detJ
                    
                    com2 =  ((np.matmul(Nj.T,dNdx)*c)+(np.matmul(np.matmul(Nj.T,np.matmul(dNdx, T[EL2NOD[iel,:]]) ),N.T))*cprime)*self.v*detJ
                    
                    Ael     = Ael + com1 + com2
                    
                    # 5. assemble right-hand side, no source terms, just here for completeness
                    Rhs_el     = Rhs_el + np.zeros(self.nnodel)
                
                # assemble coefficients
                I[iel,:]  =  (EL2NOD[iel,:]*np.ones((self.nnodel,1), dtype=int)).T.reshape(self.nnodel*self.nnodel)
                J[iel,:]  =  (EL2NOD[iel,:]*np.ones((self.nnodel,1), dtype=int)).reshape(self.nnodel*self.nnodel)
                K[iel,:]  =  Ael.reshape(self.nnodel*self.nnodel)
                
                Rhs_all[EL2NOD[iel,:]] += Rhs_el
    
            A_all = csr_matrix((K.reshape(self.nel*self.nnodel*self.nnodel),(I.reshape(self.nel*self.nnodel*self.nnodel),J.reshape(self.nel*self.nnodel*self.nnodel))),shape=(self.nnod,self.nnod))
            
            F = self.get_Residual(T) - self.get_F()
            
            u_dirichlet =np.zeros(totalDofs)
            u_dirichlet[leftDofs]=0.0
            
            u_neumann = np.zeros (totalDofs)
            u_neumann[topDofs]=1.0
            
            forceDirichlet =A_all.dot(u_dirichlet) 
            forceNeumann= np.ones(np.size(u_neumann))*0.0
            forceNeumann[botDofs] = 0.0
            
    
            finalForce = F - forceDirichlet - forceNeumann
            
            A_all[leftDofs,:]=0.0
            A_all[:,leftDofs]=0.0
            A_all[leftDofs,leftDofs]=1.0
            
            F[leftDofs]=0.0;
            dResidual = np.sqrt(np.matmul(F,F))
            
            nT = spsolve(A_all,F)
            T[:,0] = T[:,0] - nT
           # print(dResidual)
            tolerance = 1e-5
            if dResidual < tolerance:
                result =T.reshape((self.ny,self.nx))
                break  
            
        return result


    def get_piecewise_grad(self, Lambda):
        '''
        Similar structure as get_F()
        Only use the last top DOF
        
        Input: Whole Lambda value, same mesh as Forward, inside function will extract TOP DOF
        Return: The integration value projected on TOP DOF, currently return directly lambda for debugging
        '''
        GCOORD = self.get_gCoord()
        EL2NOD = self.get_EL2NOD()
        nnodel_t = 2
        nnod_t    = self.nx 
        nel_t     = self.nex    ###changed
        GCOORD_t  = np.zeros((nnod_t,1)) ###changed
        
        #Global coordinates top surface
        id_t = 0
        for i in range(0,self.nx):
                GCOORD_t[id_t] = i*self.dx 
                #GCOORD_t[id_t,1] = i*dy
                id_t           = id_t + 1
            
        #Element to nodes connectivity
        EL2NOD_t = np.zeros((nel_t, nnodel_t), dtype=int)
        for iel_t in range(0, nel_t):
            EL2NOD_t[iel_t,:] = [iel_t, iel_t+1]

        #Integration points setup top
        nip_t = 2
        gauss_t = np.array([-np.sqrt(1/3),np.sqrt(1/3)]).T.copy()

        Lambda_cont = laserPolynomial(Lambda[-self.nx:])
        F = np.zeros((self.nx*self.ny))
        
        for iel_t in range (0, nel_t):
            ECOORD_t  = np.take(GCOORD_t, EL2NOD_t[iel_t,:], axis=0) #retrieve element coordinates
            
            Fel_t = np.zeros((nnodel_t))
            
            for ip_t in range(0, nip_t):
                
                #use the shape functions
                xi_t      = gauss_t[ip_t]
                N_t, dNds_t = shapes_1D(xi_t)
        
                #setup Jacobian
                Jac_t     = np.matmul(dNds_t, ECOORD_t) #element length/element size in para. space
                invJ_t    = 1/Jac_t
                detJ_t    = Jac_t[0]
            
                #get global derivatives
                
                dNdx_t    = invJ_t*dNds_t
                elid = iel_t + self.nex*(self.ney-1)
                x_global,y_global = localmaptoglobal_2D(xi_t, 1, elid)
                
            
                Fel_t = Fel_t + N_t * Lambda_cont(x_global) * detJ_t  #do not need to multiply N_t
                
            Global_id = EL2NOD[elid,:]   
            Global_top_id = Global_id[-1:-3:-1] 
            #Global_top_id = Global_id[2:4]  +1   [5,6], [6,5]
            F[Global_top_id] += Fel_t 
            
            
        #return F[-self.nx:]
        return Lambda[-self.nx:]*1/detJ_t

    def plot_T_contour(self, T):
        X = np.reshape(self.GCOORD[:,0], (self.ny,self.nx))
        Y = np.reshape(self.GCOORD[:,1], (self.ny,self.nx))
        plt.ion()
        fig1 = plt.figure()
        #plt.xlim([0, 5600])
        cp1 = plt.contourf(X, Y, T, cmap="turbo", vmin=self.vmin, vmax=self.vmax)
        plt.colorbar(cp1)
        contour_value = 1300
        plt.contour(X, Y, T, levels=[contour_value], colors='red', linestyles='solid', linewidths=2)
        plt.title('Temperature within the domain')
        plt.xlabel("x")
        plt.ylabel("y")
        aspect_ratio = (self.ly*3) / (self.lx)        
        plt.gca().set_aspect(aspect_ratio)
        plt.show()
        plt.ioff()
    