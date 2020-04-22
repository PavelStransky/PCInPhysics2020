import numpy as np
import matplotlib.pyplot as plt

import time

generator = np.random.default_rng()

def GaussianFunction(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-x**2 / 2)


def GaussianGenerator1():
    """ Central Limit Theorem - sum of 6 uniformly distributed values """
    gaussianNumber = sum(generator.random(12)) - 6
    return gaussianNumber


def GaussianGenerator2():
    """ Hit-and-miss method in a rectangle [-6,6]x[0, 1/sqrt(2 pi) ~ 0.4] """
    while True:
        x = 12 * generator.random() - 6
        y = 0.4 * generator.random()
        if y < GaussianFunction(x):
            return x


def ExecutionTimeComparison(numValues=1000000):
    startTime = time.time()

    for _ in range(numValues):
        generator.normal()
    endTime = time.time()
    print(f"Numpy normal routine: {endTime - startTime}s")

    startTime = time.time()
    for _ in range(numValues):
        GaussianGenerator1()
    endTime = time.time()
    print(f"Generator using Central Limit Theorem: {endTime - startTime}s")

    startTime = time.time()
    for _ in range(numValues):
        GaussianGenerator2()
    endTime = time.time()
    print(f"Generator using Hit-and-miss: {endTime - startTime}s")

ExecutionTimeComparison()