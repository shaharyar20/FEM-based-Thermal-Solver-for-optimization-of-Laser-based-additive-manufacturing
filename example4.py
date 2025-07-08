#-------------------------------------------------------------------------------
# Example 4
#
# Run the forward solver
#-------------------------------------------------------------------------------
from laserOptimizer import *

if __name__ == "__main__":
    
    plt.close('all')
    settings = ProblemSettings()
    
    laser = laserGaussian([30,2.5,9])
    
    coeff = projectPiecewise(settings.lx, settings.nex, laser) # [1]-nex
    plt.plot(coeff)    
    plt.show()
    
    forCase = ForwardSolver(settings, coeff[:,0])

    tarT= forCase.get_T_nonlin()
    
    forCase.plot_T(tarT)

    curT = np.zeros((settings.ny,settings.nx))
                    
    def num_gradient(params):
        '''
        Finite difference of the parameters

        '''
        h = 1e-5
        grad = np.zeros_like(params)
        
        for i in range(len(params)):
            tmp = np.copy(params)
            tmp[i] = tmp[i] + h
            fr = loss_function(ForwardSolver(settings, tmp).get_T_nonlin_nonlin(),tarT)
            #params_tmp = float(tmp - h)
            tmp[i] = tmp[i] - 2*h
            fl = loss_function(ForwardSolver(settings, tmp).get_T_nonlin_nonlin(),tarT)
            grad[i] = (fr-fl)/(2*h)
            #params[i] = tmp
        return grad
    
    invCase = InverseSolver(settings, tarT, curT)
    Lambda = invCase.get_Lambda()
    invCase.plot_Lambda()
    adjoint_gradient = forCase.get_piecewise_grad(Lambda[-settings.nx])
    
    manGrad_trapz =np.zeros((np.size(adjoint_gradient)))
    for i in range(manGrad_trapz.size):
        loss1= loss_trapz(ForwardSolver(settings, coeff[:,0]).get_T_nonlin()[0],np.zeros(np.shape(tarT)))
        coeff[i,0]+=0.0001
        loss2=loss_trapz(ForwardSolver(settings, coeff[:,0]).get_T_nonlin()[0],np.zeros(np.shape(tarT)))
        manGrad_trapz[i]=(loss2-loss1)/0.0001
        plt.plot(manGrad_trapz)
        plt.show()
   
    manGrad_gauss =np.zeros((np.size(adjoint_gradient)))
    for i in range(manGrad_gauss.size):
        loss1= loss_gauss(ForwardSolver(settings, coeff[:,0]).get_T_nonlin()[0],np.zeros(np.shape(tarT)))
        coeff[i,0]+=0.0001
        loss2=loss_gauss(ForwardSolver(settings, coeff[:,0]).get_T_nonlin()[0],np.zeros(np.shape(tarT)))
        manGrad_gauss[i]=(loss2-loss1)/0.0001
        plt.plot(manGrad_gauss)
        plt.show()

        
    plt.plot(manGrad_trapz)
    plt.plot(manGrad_gauss)
    plt.plot(-adjoint_gradient)
    
    plt.show()
    