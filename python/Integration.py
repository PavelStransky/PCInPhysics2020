""" Monte-Carlo integration """
import numpy as np
import matplotlib.pyplot as plt

generator = np.random.default_rng()


def f1(x):
    return np.exp(-x) * np.sin(x)


def f2(x):
    return np.sin(x)**2 / np.sqrt(1 + x**4)


def Integrate1D(n, Function, a, b):
    """ Basic Monte-Carlo 1D integration """
    result = 0
    for _ in range(n):
        x = (b - a) * generator.random() + a
        result += Function(x)

    result = (b - a) / n * result
    return result


def Integrate1DArray(n, Function, a, b):
    """ Monte Carlo integration via array. It requieres more memory, but it is much faster. """
    x = (b - a) * generator.random(n) + a
    result = (b - a) / n * sum(Function(x)) 
    return result


def Integral2(n):
    hits = 0
    result = 0

    for _ in range(n):
        x, y, z, w = generator.random(4)
        
        if (x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2 + (w - 0.5)**2 <= 0.25:
            hits += 1
            result += np.sin(np.sqrt(np.log(x + y + z + w + 2)))

    result = result / hits
    volume = hits / n

    print(f"I3 = {result} (number of hits: {hits}, volume of the integration region: {volume})")

    return result


def CalculateIntegrals():
    print("I1 = {}".format(Integrate1DArray(1000000, f1, 0, 2 * np.pi)))
    print("I2 = {}".format(Integrate1DArray(1000000, f2, 0, np.sqrt(10 * np.pi))))
    print("I3 = {}".format(Integral2(1000000))
