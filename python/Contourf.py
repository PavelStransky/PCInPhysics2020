import matplotlib.pyplot as plt
from matplotlib import cm           # Colour maps for the contour graph
import numpy as np

def f(x, y):
    return x**4 - 2*x**2 + x + y**2

step = 0.025
numContours = 30                    # Number of contours in the graph

x = y = np.arange(-2.0, 2.01, step) # Range of x and y values for the graph

X, Y = np.meshgrid(x, y)            # Grid for calculating values of the function
Z = f(X, Y)

plt.contourf(X, Y, Z, numContours, cmap=cm.hot)  
plt.colorbar()                      # Legend for the contour graph
plt.show()
