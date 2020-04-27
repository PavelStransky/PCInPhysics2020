import random
import numpy as np
import matplotlib.pyplot as plt

# Switch on the interactive mode (in order to plot more random walks into one graph)
# - In Spyder you have to write the command "%matplotlib auto" into the REPL first
plt.ion()


def RandomDirection2D():
    """ Generates random direction in a 2D plane """
    phi = random.uniform(0, 2 * np.pi)
    return np.array([np.cos(phi), np.sin(phi)])
    

def RandomWalk2D(numSteps=10000, stepSize=1, boxSize=100, initialCondition=[0,0]):
    """ Generates and plots a random walk in a 2D plane.
        If the trajectory leaves the given box, the calculation is interrupted.
    """    
    position = np.array(initialCondition)
    path = [position]

    box = np.array([boxSize, boxSize])

    for i in range(0, numSteps):
        position = position + stepSize * RandomDirection2D()# for some unknown reason command
                                                            # position += RandomStep2D(stepSize)
                                                            # doesn't work

        if not np.allclose(position, (position + box) % (2*box) - box):
            break                                           # Check whether we are within the given box
        
        path.append(position)

    path = np.array(path)
    plt.plot(path[:,0], path[:,1], label="N=%d" % len(path))
    plt.legend()
    plt.show()

    return path


def RandomWalk2DInteractive(numSteps=10000, stepSize=1, boxSize=100, initialCondition=[0,0]):
    """ Generates and interactively plot a random walk in a 2D plane.
        If the trajectory leaves the given box, the calculation is interrupted.
    """    
    position = np.array(initialCondition)
    path = [position]

    box = np.array([boxSize, boxSize])    

    plt.xlim(-boxSize, boxSize)
    plt.ylim(-boxSize, boxSize)

    for i in range(0, numSteps):
        newPosition = position + stepSize * RandomDirection2D()

        if not np.allclose(newPosition, (newPosition + box) % (2*box) - box):
            break                                       # Check whether we are within the given box

        path.append(newPosition)

        plt.plot([position[0], x], [position[1], y], 'blue')
        plt.show()
        plt.pause(0.1)

        position = newPosition

    path = np.array(path)
    return path
