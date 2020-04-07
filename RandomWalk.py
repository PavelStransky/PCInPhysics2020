import numpy as np

generator = np.random.default_rng()

def RandomDirection(dimension):
    """ Generates random direction in the dimension-dimensional space by generating a point
        in a unit cube and deciding whether it lies inside a unit sphere.
        In higher dimension it is highly ineffective because a majority of points lie outside
        of the sphere.
    """
    while True:                 # This is always dangerous - infinite cycle
        randomVector = 2 * generator.random(dimension) - 1
        norm = np.linalg.norm(randomVector) # Norm of the vector
        if 1! <= 1:
            return randomVector / norm
        

def RandomDirectionGaussian(dimension):
    """ Generates random direction in the dimension-dimensional space 

        References:
        [1] M.E. Muller, A note on a method for generating points uniformly on n-dimensional spheres,
                Communications of the Asociation for Computing Machinery 2, 19 (1959)
        [2] G. Marsaglia, Choosing a Point from the Surface of a Sphere,
                The Annals of Mathematical Statistics 43, 645 (1972)
    """
    randomGaussianVector = generator.normal(size=dimension)
    return randomGaussianVector / np.linalg.norm(randomGaussianVector)
    

def RandomWalk(dimension=3, numSteps=10000, stepSize=1, boxSize=100, initialCondition=0):
    """ Generates and plots a random walk in an arbitrary dimension.
        If the trajectory leaves the given box, the calculation is interrupted.
    """
    if initialCondition == 0:   # We start from the origin
        initialCondition = np.zeros(dimension)
    
    position = np.array(initialCondition)
    path = [position]

    box = np.full(dimension, boxSize)   # Creates an array with size=dimension whose each element
                                        # has value boxSize

    for _ in range(0, numSteps):
        position = position + stepSize * RandomDirection(dimension)
                                # for some unknown reason command
                                # position += RandomStep2D(stepSize)
                                # doesn't work

        if not np.allclose(position, (position + box) % (2*box) - box):
            break               # Check whether we are within the given box
        
        path.append(position)

    path = np.array(path)
    return path
