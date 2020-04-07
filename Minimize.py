import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm           # Colour maps for the contour graph

from scipy.optimize import minimize

#from RandomWalkMinimize import *
from MetropolisMinimize import *  

def f(X):
    """ Quadratic test function """ 
    x, y = X
    return x*x + y*y


def g(X, parameters=(1, 100)):
    """ Rosenbrock test function 
        https://en.wikipedia.org/wiki/Rosenbrock_function
    """
    x, y = X
    a, b = parameters
    return (a - x)**2 + b * (y - x * x)**2


def h(X):
    """ 4D test function """
    s, t, u, v = X
    sum2 = s*s + t*t + u*u + v*v
    if sum2 > 2:
        return float("inf")
    return 0.25 * sum2 - 0.5 * ((s*s + t*t) * (2 - sum2) + (s*u - t*v)**2) + 0.5 * s * np.sqrt(2 - sum2)

def r(X, parameters=(1,)):
    """ Double-well test function """
    x, y = X
    a, = parameters
    return x**4 - 2 * x*x + a * x + y*y


def ShowGraph(Function, paths=(), boxSize=1, numContours=100):
    """ Plots a contour graphs of the give 2D function.

    Arguments:
    Function -- function to plot
    parameters -- additional parameters of the function
    paths -- curves to plot in the graph
    boxSize -- limits of the x, y range
    numContours -- number of contours in the graph
    """
    x = y = np.linspace(-boxSize, boxSize, 100) # Range of x and y values for the graph

    X, Y = np.meshgrid(x, y)                    # Grid for calculating values of the function
    Z = Function([X, Y])

    plt.figure(figsize=(8,6))
    plt.contourf(X, Y, Z, numContours, cmap=cm.hot)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label=Function.__name__ + '(x,y)')  # Legend for the contour graph

    for path in paths:
        plt.plot(path[:,0], path[:,1])  
        plt.scatter(path[[0,-1],0], path[[0,-1],1])  # Indicate the initial and final points of the path
    plt.tight_layout()
    plt.show()


def MultiplePaths(Function, numPaths=10, initialConditionBox=2, **kwargs):
    """ Shows graph with a given number of random walk minimization paths. """
    paths = []

    for _ in range(0,numPaths):
        paths.append(Minimize(Function, initialConditionBox=initialConditionBox, **kwargs))

    ShowGraph(Function, paths, boxSize=1.2*initialConditionBox)


def MultiplePathsAdaptive(Function, numPaths=10, initialConditionBox=2, **kwargs):
    """ Shows graph with a given number of random walk minimization paths (method with adaptive step). """
    paths = []

    for _ in range(0, numPaths):
        paths.append(MinimizeAdaptive(Function, initialConditionBox=initialConditionBox, **kwargs))

    ShowGraph(Function, paths, boxSize=1.2*initialConditionBox)
