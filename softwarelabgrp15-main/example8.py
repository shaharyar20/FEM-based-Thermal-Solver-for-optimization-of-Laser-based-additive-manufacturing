#-------------------------------------------------------------------------------
# Example 8
#
# Gradient Algorithm Comparison
#-------------------------------------------------------------------------------
from laserOptimizer import *
import psutil
import time

start_time = time.time()
process = psutil.Process()

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


# fndGrad = np.zeros((np.size(pst.neLaser+1)))
# for i in range(fndGrad.size):
#     loss1 = loss_gauss(ForwardSolver(pst, coeff[:,0]).get_T()[0],np.zeros(np.shape(tarT)))
#     coeff[i,0] += 0.000001
#     loss2 = loss_gauss(ForwardSolver(pst, coeff[:,0]).get_T()[0],np.zeros(np.shape(tarT)))
#     fndGrad[i] = (loss2 - loss1) / 0.000001



max_memory_used = process.memory_info().peak_wset / (1024 ** 2)  # in megabytes
print(f"Max memory used by the process: {max_memory_used:.2f} MB")

end_time = time.time()
run_time = end_time - start_time
print(f"Program execution time: {run_time:.2f} seconds")