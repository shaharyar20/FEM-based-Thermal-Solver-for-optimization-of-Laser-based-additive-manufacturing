#-------------------------------------------------------------------------------
# Example 3
#
# Run the optimization
#-------------------------------------------------------------------------------
from laserOptimizer import *
from scipy.optimize import minimize

if __name__ == "__main__":
    
    plt.close('all')
    settings = ProblemSettings()
    
    laser = laserGaussian([30,2.5,9])
    
    coeff = projectPiecewise(settings.lx, settings.nex, laser) # [1]-nex
    plt.plot(coeff)    
    plt.show()
    
    forCase = ForwardSolver(settings, coeff[:,0])

    tarT, tarTarray= forCase.get_T()
    
    forCase.plot_T(tarT)

    curT = np.zeros((settings.ny,settings.nx))

    
    invCase = InverseSolver(settings, tarT, curT)
    Lambda = invCase.get_Lambda()
    invCase.plot_Lambda()
    adjoint_gradient = forCase.get_piecewise_grad(Lambda[-settings.nx:-1])
    
    def manGrad_trapz(coeff):
        manGrad_trapz =np.zeros((np.size(adjoint_gradient)))
        for i in range(manGrad_trapz.size):
            loss1= loss_trapz(ForwardSolver(settings, coeff[:,0]).get_T()[0],np.zeros(np.shape(tarT)))
            coeff[i,0]+=0.0001
            loss2=loss_trapz(ForwardSolver(settings, coeff[:,0]).get_T()[0],np.zeros(np.shape(tarT)))
            manGrad_trapz[i]=(loss2-loss1)/0.0001
            plt.plot(manGrad_trapz)
            plt.show()
        return manGrad_trapz

    result = minimize(loss_trapz, coeff, jac = manGrad_trapz, method='L-BFGS-B', tol=1e-6 ,options={'disp': True})
    