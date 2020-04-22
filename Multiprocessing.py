import numpy as np
import time

from random import *
from multiprocessing import Pool, Process, Value

def Integral1(n):
    a = 0
    b = 2

    result = 0
    for _ in range(n):
        x = uniform(a, b)
        result += np.sin(np.pi * x)**2

    return (b - a) * result / n

def Integral1Pool(n, numProcesses):
    startTime = time.time()
    with Pool(processes=numProcesses) as pool:
        ns = [n for i in range(numProcesses)]
        results = pool.map(Integral1, ns)

    endTime = time.time()
    average = sum(results) / len(results)

    print(f"Výsledek: {average}, počet vláken: {numProcesses}, doba výpočtu: {endTime - startTime}s")

def Integral1Process(n, numProcesses):
    vlakna = []
    vysledek = []

    for i in range(pocetVlaken):
        v = Value('d', 0)
        p = Process(target=Integral2V, args=(n,v))
        p.start()
        vlakna.append(p)
        vysledek.append(v)

    for p in vlakna:
        p.join()

    return vysledek

if __name__ == '__main__':
    Integral1Pool(1000000, 8)
