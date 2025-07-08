# -------------------------------------------------------------------------------
# optimization_test
#
# Test optimization loop
# -------------------------------------------------------------------------------
from laserOptimizer import *
from scipy.optimize import minimize

if __name__ == "__main__":

    plt.close('all')
    settings = ProblemSettings()

# Retrieve laser coeff from gauss
    laser = laserGaussian([16000, 0.01, 0.05])
    coeff = projectPiecewise(laser)
    plt.title('Laser profile on top surface')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(coeff)
    plt.show()
    
# Set up forwardsolver object and retrieve temperature to achieve (tarT)
    forSolver = ForwardSolver(settings, coeff[:, 0])
    tarT = forSolver.get_T()[0]
    forSolver.plot_T_contour(tarT)

# Set up inversesolver object
    curT = np.zeros_like(tarT)

    invSolver = InverseSolver(settings, tarT, 0,curT)
    Lambda = invSolver.get_Lambda()
    invSolver.plot_Lambda()
    adjoint_gradient = forSolver.get_piecewise_grad(Lambda)
    #
    plt.plot(adjoint_gradient)
    
    plt.show()

# Manually retrieve gradient through finite difference
    #manGrad_gauss = np.zeros((np.size(adjoint_gradient)))
    '''
    def fd_gradient(manGrad_gauss):
        for i in range(manGrad_gauss.size):
            coeff[i, 0] = coeff[i, 0] + 0.0001
            loss1 = loss_gauss(ForwardSolver(settings, coeff[:, 0]).get_T()[
                               0], np.zeros(np.shape(tarT)))
            coeff[i, 0] = coeff[i, 0] - 0.0002
            loss2 = loss_gauss(ForwardSolver(settings, coeff[:, 0]).get_T()[
                               0], np.zeros(np.shape(tarT)))
            manGrad_gauss[i] = (loss1-loss2)/0.0002
        return manGrad_gauss
    '''
    def fd_gradient(params):
        manGrad_gauss = np.zeros_like(params)
        
        for i in range(manGrad_gauss.size):
            params[i] = params[i] + 0.0001
            loss1 = loss_gauss(ForwardSolver(settings, params[:]).get_T()[
                               0], tarT)
            params[i] = params[i] - 0.0002
            loss2 = loss_gauss(ForwardSolver(settings, params[:]).get_T()[
                               0], tarT)
            manGrad_gauss[i] = (loss1-loss2)/0.0002
        return manGrad_gauss
    
#plt.plot(manGrad_gauss)
#plt.show()

# Set up loss function tarT
    def loss (params) :
        res = loss_gauss(ForwardSolver(settings, params[:]).get_T()[0] , tarT)
        print("loss is: ", res)
        return res

    def gradient_wrapper(params):
        '''
        Wrapper for processing: params -> Temperature -> Lambda -> Gradient
        tarT is defined above
        '''
        currentT = ForwardSolver(settings, params[:]).get_T()[0]
        invSolver = InverseSolver(settings, tarT, 0, currentT)
        Lambda = invSolver.get_Lambda()
        #invSolver.plot_Lambda()
        grad_4opti = computeGradient(invSolver)
        #grad_4opti = projectGrad(grad_4opti)
        #print("Gradient: ", np.linalg.norm(grad_4opti))
        #plt.plot(grad_4opti)
        return -grad_4opti
        
# Run Optimization loop
    x0 = np.zeros_like(coeff)
    #x0 = coeff ##test
    result = minimize(loss, x0, jac=gradient_wrapper,
                      method='L-BFGS-B', tol=1e-8, options={'disp': True})
    
    plt.title('Laser profile on top surface')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(coeff)
    plt.plot(result.x)
    plt.show()
                      
    forSolver_post = ForwardSolver(settings, result.x)
    post_T = forSolver_post.get_T()[0]
    forSolver.plot_T_contour(post_T)