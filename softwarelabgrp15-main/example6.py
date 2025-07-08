#-------------------------------------------------------------------------------
# Example 6
#
# Lambda Convergence Test
#-------------------------------------------------------------------------------
from laserOptimizer import *

pst = ProblemSettings()

laser = laserGaussian([8000, 0.005, 0.025])
coeff = projectPiecewise(laser)

forSolver = ForwardSolver(pst, coeff[:, 0])
tarT,tarTarr = forSolver.get_T()

curT = np.zeros_like(tarT)

invCase = InverseSolver(pst, tarT, tarTarr, curT)
lbd = invCase.get_Lambda()

print(type(lbd))
print(f"The mean value of lambda is {np.mean(lbd)}")
print(f"The resolution {pst.lx/pst.nx}")