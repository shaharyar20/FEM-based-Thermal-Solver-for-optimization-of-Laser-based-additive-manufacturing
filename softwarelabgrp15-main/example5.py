#-------------------------------------------------------------------------------
# Example 5
#
# Temperature Convergence Test
#-------------------------------------------------------------------------------
from laserOptimizer import *

pst = ProblemSettings()

laser = laserGaussian([8000, 0.005, 0.025])
coeff = projectPiecewise(laser)

forSolver = ForwardSolver(pst, coeff[:, 0])
tarT = forSolver.get_T()[0]
#forSolver.plot_T(tarT)

print(type(tarT))
print(f"The mean value of tarT is {np.mean(tarT)}")
print(f"The resolution {pst.lx/pst.nx}")