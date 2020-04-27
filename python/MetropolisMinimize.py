import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm           # Colour maps for the contour graph

from RandomWalk import RandomDirectionGaussian

generator = np.random.default_rng()
   
def Minimize(Function, dimension=2, temperature=1, stepSize=0.01, initialCondition=0, initialConditionBox=1, maxSteps=10000):
    """ Simple minimization method using random walk and Metropolis algorithm.

    Arguments:
    Function -- function to minimize
    dimension -- dimensionality of the function
    temperature -- temperature of the Metropolis algorithm
    initialCondition -- a point of initialconditions; if 0, initial conditions chosen randomly from a box of size given by initialConditionBox
    maxSteps -- total number of steps
    """
    if initialCondition == 0:       # Random initial condition within box of size given by initialConditionBox
        initialCondition = initialConditionBox * (2 * generator.random(dimension) - 1)
    
    position = np.array(initialCondition)
    f = Function(position)

    path = [position]

    numSteps = 0                    # Total number of steps

    while numSteps < maxSteps:
        newPosition = position + stepSize * RandomDirectionGaussian(dimension)
        newf = Function(newPosition)
        C = np.exp((f - newf) / temperature)    # Boltzmann coefficient (for step down C > 1)

        if C > generator.random():
            position = newPosition
            f = newf

            path.append(position)

        numSteps += 1

    print("Minimum = {}, function value = {}, steps = {}".format(position, f, numSteps))

    return np.array(path)


def MinimizeAdaptive(Function, dimension=2, initialTemperature=1, initialStepSize=1, finalStepSize=1E-6, initialCondition=0, initialConditionBox=1, numStepsChange=100):
    """ Minimization method using random walk and Metropolis algorithm with adaptive temperature.

    Arguments:
    Function -- function to minimize
    dimension -- dimensionality of the function
    inititalTemperature -- initital temperature of the Metropolis algorithm
    initialStepSize, finalStepSize -- 
    initialCondition -- a point of initialconditions; if 0, initial conditions chosen randomly from a box of size given by initialConditionBox
    numStepsChange -- number of steps to decrease the step size and temperature by factor 1/2
    """
    if initialCondition == 0:       # Random initial condition within box of size given by initialConditionBox
        initialCondition = initialConditionBox * (2 * generator.random(dimension) - 1)
    
    position = np.array(initialCondition)
    f = Function(position)

    path = [position]

    numSteps = 0                    # Total number of steps
    stepSize = initialStepSize
    temperature = initialTemperature

    while stepSize > finalStepSize:
        newPosition = position + stepSize * RandomDirectionGaussian(dimension)
        newf = Function(newPosition)
        C = np.exp((f - newf) / temperature)    # Boltzmann coefficient (for step down C > 1)

        if C > generator.random():
            position = newPosition
            f = newf

            path.append(position)
    
        numSteps += 1

        if numSteps % numStepsChange == 0:
            stepSize *= 0.5
            temperature *= 0.5

    print("Minimum = {}, function value = {}, steps = {}".format(position, f, numSteps))

    return np.array(path)