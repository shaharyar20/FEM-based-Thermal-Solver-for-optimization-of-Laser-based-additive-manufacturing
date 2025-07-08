# -------------------------------------------------------------------------------
# gradient_test
#
# Test the gradient value
# -------------------------------------------------------------------------------
from laserOptimizer import *
from scipy.optimize import minimize

if __name__ == "__main__":

    plt.close('all')
    settings = ProblemSettings()
    #settings.setElements(30, 10)

# Retrieve laser coeff from gauss
    laser = laserGaussian([8000, 0.005, 0.025])
    coeff = projectPiecewise(laser)
    # plt.plot(coeff)
    # plt.show()

# Set up forwardsolver object and retrieve temperature to achieve (tarT)
    forSolver = ForwardSolver(settings, coeff[:, 0])
    tarT = forSolver.get_T()[0]
    forSolver.plot_T(tarT)

# Set up inversesolver object
    curT = np.zeros_like(tarT)

    invSolver = InverseSolver(settings, tarT, 0,curT)
    Lambda = invSolver.get_Lambda()
    invSolver.plot_Lambda()
    adjoint_gradient = computeGradient(invSolver)
    #adjoint_gradient = projectGrad(adjoint_gradient)
    plt.plot(adjoint_gradient)
    # plt.show()

# Manually retrieve gradient

    manGrad_gauss =np.zeros((np.size(adjoint_gradient)))
    
    for i in range(manGrad_gauss.size):
        loss1= loss_gauss(ForwardSolver(settings, coeff[:,0]).get_T()[0],np.zeros(np.shape(tarT)))
        coeff[i,0]+=0.000001
        loss2=loss_gauss(ForwardSolver(settings, coeff[:,0]).get_T()[0],np.zeros(np.shape(tarT)))
        manGrad_gauss[i]=(loss2-loss1)/0.000001
        
    plt.plot(manGrad_gauss)
    #loss1= loss_gauss(ForwardSolver(settings, coeff[:,0]).get_T()[0],np.zeros(np.shape(tarT)))
    #print(loss1)
    
    
    
    plt.show()



