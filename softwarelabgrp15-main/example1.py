#-------------------------------------------------------------------------------
# Example 1
#
# Run the inverse solver
#-------------------------------------------------------------------------------
from laserOptimizer import *

if __name__ == "__main__":
    
    pst = ProblemSettings()

    forCase = ForwardSolver(pst, [20,2.5,0])

    tarT, tarTarray = forCase.get_T()

    forCase.plot_T(tarT)

    invCase = InverseSolver(pst, tarT, tarTarray) # In this case, curT is set to be zero
    
    invCase.plot_Lambda()

