import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm           # Colour maps for the contour graph

from RandomWalk import RandomDirectionGaussian

generator = np.random.default_rng()
   
def Minimize(Function, dimension=2, stepSize=0.01, initialCondition=0, initialConditionBox=2, maxFailedSteps=100):
    """ Simple minimization method using random walk.

    Arguments:
    Function -- function to minimize
    dimension -- dimensionality of the function
    initialCondition -- a point of initialconditions; if 0, initial conditions chosen randomly from a box of size given by initialConditionBox
    maxFailedSteps -- number of steps to stop the calculation
    """
    if initialCondition == 0:       # Random initial condition within box of size given by initialConditionBox
        initialCondition = initialConditionBox * (2 * generator.random(dimension) - 1)
    
    position = np.array(initialCondition)
    f = Function(position)

    path = [position]

    failedSteps = 0                 # Number of steps in which we haven't moved (criterion to stop the minimization) 
    numSteps = 0                    # Total number of steps

    while failedSteps < maxFailedSteps:
        newPosition = position + stepSize * RandomDirectionGaussian(dimension)
        newf = Function(newPosition)

        if newf < f:
            position = newPosition
            f = newf

            path.append(position)

            failedSteps = 0

        else:
            failedSteps += 1        # Step up 

        numSteps += 1

    print("Minimum = {}, function value = {}, steps = {}".format(position, f, numSteps))

    return np.array(path)


def MinimizeAdaptive(Function, dimension=2, initialStepSize=0.1, finalStepSize=1E-6, initialCondition=0, initialConditionBox=2, maxFailedSteps=100):
    """ Minimization method using random walk and adaptive step size

    Arguments:
    Function -- function to minimize
    dimension -- dimensionality of the function
    initialCondition -- a point of initialconditions; if 0, initial conditions chosen randomly from a box of size given by initialConditionBox
    maxFailedSteps -- number of unsuccessful steps to decrease the step size by factor 1/2
    """
    if initialCondition == 0:       # Random initial condition within box of size given by initialConditionBox
        initialCondition = initialConditionBox * (2 * generator.random(dimension) - 1)
    
    position = np.array(initialCondition)
    f = Function(position)

    path = [position]

    failedSteps = 0                 # Number of steps in which we haven't moved (criterion to stop the minimization) 
    numSteps = 0                    # Total number of steps

    stepSize = initialStepSize

    while stepSize > finalStepSize:
        newPosition = position + stepSize * RandomDirectionGaussian(dimension)
        newf = Function(newPosition)

        if newf < f:
            position = newPosition
            f = newf

            path.append(position)

            failedSteps = 0

        else:
            failedSteps += 1        # Step up

            if failedSteps > maxFailedSteps:
                failedSteps = 0
                stepSize *= 0.5     # Step size reduced to half 

        numSteps += 1

    print("Minimum = {}, function value = {}, steps = {}".format(position, f, numSteps))

    return np.array(path)
