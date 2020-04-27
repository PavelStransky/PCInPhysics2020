import numpy as np
import matplotlib.pyplot as plt

generator = np.random.default_rng()

def VolumeBall(tries=1000000, dimension=3):
    """ Volume of the dimension-dimensional unit ball calculated by Monte-Carlo Hit-And-Miss method """
    hits = 0
    
    for _ in range(tries):
        # The same code snippet as in RandomWalk in an arbitrary dimension
        randomVector = 2 * generator.random(dimension) - 1
        norm = np.linalg.norm(randomVector) # Norm of the vector
        
        if norm <= 1:
            hits += 1

    volumeCube = 2**dimension
    volumeBall = hits / tries * volumeCube
    error = np.sqrt(hits) / tries * volumeCube

    print(f"V{dimension}={volumeBall} += {error} (Počet zásahů: {hits})")

    return volumeBall, error


def PlotVolumes(tries=1000000, dimensions=range(1, 16)):
    """ Graph of volumes of various dimensional balls (including error) """
    result = np.asarray([VolumeBall(tries, dimension) for dimension in dimensions])

    volumes = result[:, 0]
    errors = result[:, 1]

    plt.errorbar(dimensions, volumes, yerr=errors)
    plt.title("Objem $d$-rozměrné jednotkové koule")
    plt.xlabel("$d$")
    plt.ylabel("$V$")
    plt.show()

VolumeBall(dimension=15)