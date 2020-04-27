import numpy as np
import matplotlib.pyplot as plt 

import time

from multiprocessing import Pool, Process, Value
from Integration import *

generator = np.random.default_rng()


def Integrate1DPool(p, n, Function, a, b):
    """ Using Pool and map to perform the Monte-Carlo integration """
    with Pool(processes=p) as pool:
        args = ((n, Function, a, b),) * p
        # Copies p-times the given n-tuple. The same as 
        # args = [(n, Function, a, b) for _ in range(p)]
        results = pool.starmap(Integrate1D, args)

    average = sum(results) / len(results)
    return average


def PlotIntegrate1DDuration(processes=range(1, 17), n=1000000, Function=f1, a=0, b=2*np.pi):
    def Duration(p):
        startTime = time.time()
        result = Integral1DPool(p, n, Function, a, b)
        duration = time.time() - startTime
        print(f"I({Function.__name__}) = {result} (Doba výpočtu v {p} vláknech: {duration})")
        return duration

    durations = [Duration(p) for p in processes]
    
    plt.plot(processes, durations)
    plt.title("Celkový čas výpočtu")
    plt.xlabel(r"$p$")
    plt.ylabel(r"$T [s]$")
    plt.show()

    plt.plot(processes, np.array(durations) / np.array(processes))
    plt.title("Čas výpočtu na vlákno")
    plt.xlabel(r"$p$")
    plt.ylabel(r"$t [s]$")
    plt.show()


def Integrate1DP(result, *args, **kwargs):
    """ Wrapper for function Integrate1D to Process parallelization """
    result.value = Integrate1D(*args, **kwargs)


def Integrate1DProcess(p, n, Function, a, b):
    """ Using Process to perform Monte-Carlo integration """
    processes = []
    results = []

    for _ in range(p):
        result = Value('d', 0)
        process = Process(target=Integrate1DP, args=(result, n, Function, a, b))
        process.start()
        processes.append(process)
        results.append(result)

    for process in processes:
        process.join()

    results = [result.value for result in results]
    return sum(results)

if __name__ == "__main__":
    print(Integrate1DPool(8, 100000, f1, 0, 2 * np.pi))
    PlotIntegral1DDuration()
    print(Integrate1DProcess(8, 100000, f1, 0, 2 * np.pi))