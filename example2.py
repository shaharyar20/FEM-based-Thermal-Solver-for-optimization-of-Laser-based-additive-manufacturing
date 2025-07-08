#-------------------------------------------------------------------------------
# Example 1
#
# Run the forward solver
#-------------------------------------------------------------------------------
from laserOptimizer import *

if __name__ == "__main__":
    
    pst = ProblemSettings()
    
    tarLaser = 9 * np.ones([pst.nx,1])

    forCase = ForwardSolver(pst, tarLaser)

    forCase.get_T_nonlin()

    forCase.plot_T()