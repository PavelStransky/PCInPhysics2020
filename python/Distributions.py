import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

from Histogram import *
from GaussianGenerators import GaussianGenerator1, GaussianGenerator2, GaussianFunction

generator = np.random.default_rng()

def Plot(data, title="", DistributionFunction=None, **kwargs):
    """ Plots data and compares them with theoretic distributionFunction, if specified """
    x, histogram = Histogram(data, **kwargs)

    plt.tight_layout()
    plt.plot(x, histogram, label="Histogram")
    plt.title(title)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\rho$")
    plt.ylim(0)

    if DistributionFunction is not None:
        plt.plot(x, DistributionFunction(x), label="Hustota pravděpodobnosti")
        plt.legend()

    plt.show()


def Uniform(numValues=100000):
    data = generator.random(numValues)

    def DistributionFunction(x):
        return np.where((x > 0) & (x <= 1), 1, 0)

    Plot(data, r"Rovnoměrné rozdělení na $\langle0,1\rangle$", DistributionFunction, normalize=True)


def Sum2Uniform(numValues=100000):
    data = generator.random(numValues) + generator.random(numValues)

    def DistributionFunction(x):
        return np.where(x < 0, 0, np.where(x < 1, x, np.where(x < 2, 2 - x, 0)))

    Plot(data, "Součet dvou rovnoměrně rozdělených čísel", DistributionFunction, normalize=True)


def SumUniform(num=2, numValues=100000):
    data = generator.random(numValues)

    for _ in range(num - 1):
        data += generator.random(numValues)

    def DistributionFunction(x):
        sigma = np.sqrt(num / 12)
        return GaussianFunction((x - num / 2) / sigma) / sigma

    Plot(data, f"Součet {num} rovnoměrně rozdělených čísel", DistributionFunction, normalize=True)


def Gaussian1(numValues=1000000):
    data = [GaussianGenerator1() for _ in range(numValues)]

    Plot(data, "Gaussovské rozdělení (Suma 12)", GaussianFunction, normalize=True) 


def Gaussian2(numValues=1000000):
    data = [GaussianGenerator2() for _ in range(numValues)]

    Plot(data, "Gaussovské rozdělení (Hit-And-Miss)", GaussianFunction, normalize=True) 


def Poisson(numBins=100000, numValues=1000000):
    data = generator.random(numValues)
    histogram = Histogram(data, numBins=numBins)[1]

    minValue = 0
    maxValue = max(histogram)

    l = numValues / numBins
    def DistributionFunction(x):
        return l**x / gamma(x + 1) * np.exp(-l)

    Plot(histogram, "Poissonovo rozdělení", DistributionFunction, normalize=True, minValue=minValue, maxValue=maxValue, numBins=int(maxValue+1))


def FGenerator(size=1):
    F = generator.random(size=size)
    return np.tan(np.pi / 2 * (2*F - 1))


def Cauchy(numValues=100000):
    data = FGenerator(numValues)

    def DistributionFunction(x):
        return 1 / (np.pi * (1 + x**2))

    Plot(data, "Cauchyho rozdělení", DistributionFunction, normalize=True, minValue=-10, maxValue=10, numBins=500)

Uniform()
Sum2Uniform()

SumUniform(3)
SumUniform(4)
SumUniform(20)

Gaussian1()
Gaussian2()
Poisson()
Cauchy()