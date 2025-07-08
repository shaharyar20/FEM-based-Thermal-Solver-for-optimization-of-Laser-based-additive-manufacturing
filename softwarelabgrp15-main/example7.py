#-------------------------------------------------------------------------------
# Example 7
#
# Gradient Convergence Test
#-------------------------------------------------------------------------------
from laserOptimizer import *

def get_theta(vct):
    '''
    Calculate the angle between the given vector and [1,...,1]
    '''
    target_vector = np.ones_like(vct)
    dot_product = np.dot(vct, target_vector)
    angle_cosine = dot_product / (np.linalg.norm(vct) * np.linalg.norm(target_vector))
    return np.arccos(angle_cosine)

def get_theta_degrees(vct):
    '''
    Calculate the angle between the given vector and [1,...,1] in degrees
    '''
    target_vector = np.ones_like(vct)
    dot_product = np.dot(vct, target_vector)
    angle_cosine = dot_product / (np.linalg.norm(vct) * np.linalg.norm(target_vector))
    angle_radians = np.arccos(angle_cosine)
    return np.degrees(angle_radians)

pst = ProblemSettings()

laser = laserGaussian([8000, 0.005, 0.025])
coeff = projectPiecewise(laser)

forSolver = ForwardSolver(pst, coeff[:, 0])
tarT,tarTarr = forSolver.get_T()
curT = np.zeros_like(tarT)

invSolver = InverseSolver(pst, tarT, tarTarr, curT)
lbd = invSolver.get_Lambda()

adjGrad = computeGradient(invSolver)

# fndGrad = np.zeros((np.size(adjGrad)))
# for i in range(fndGrad.size):
#     loss1= loss_gauss(ForwardSolver(pst, coeff[:,0]).get_T()[0],np.zeros(np.shape(tarT)))
#     coeff[i,0] += 0.000001
#     loss2=loss_gauss(ForwardSolver(pst, coeff[:,0]).get_T()[0],np.zeros(np.shape(tarT)))
#     fndGrad[i] = (loss2 - loss1) / 0.000001



# print(f"The theta of finite difference gradient is {get_theta(fndGrad)}")
print(f"The theta of adjoint gradient is {get_theta(adjGrad)}")

# print(f"The theta of finite difference gradient is {get_theta_degrees(fndGrad)}")
print(f"The theta of adjoint gradient is {get_theta_degrees(adjGrad)}")

print(f"The resolution {pst.lx/pst.nx}")