import numpy as np
import matplotlib.pyplot as plt

import time

import Sound


def FourierTransform(signal):
    N = len(signal)

    argument = -2j * np.pi / N * np.arange(N)

    t = time.time()
    H = np.array([sum(signal * np.exp(k * argument)) for k in range(N)]) / N
    print(time.time() - t)

    return H


def InverseFourierTransform(signal):
    N = len(signal)

    argument = 2j * np.pi / N * np.arange(N)

    t = time.time()
    H = np.array([sum(signal * np.exp(k * argument)) for k in range(N)])
    print(time.time() - t)

    return H


def AmplitudeSpectrum(ft, fs):
    N = len(ft)
    powerSpectrum = 2 * np.abs(ft)
    powerSpectrum[0] = 0.5 * powerSpectrum[0]
    return np.linspace(0, fs // 2, N // 2), powerSpectrum[0:N // 2]


def TestSignal(N=2000):
    x = np.linspace(0, 1, N)
    fs = N
    y = 0.1 * np.sin(440 * 2 * np.pi * x) + 0.2 * np.sin(5 / 4 * 440 * 2 * np.pi * x) + 0.3 * np.sin(3 / 2 * 440 * 2 * np.pi * x)
    Sound.Play(y, fs)

    ft = FourierTransform(y)
    plt.plot(*AmplitudeSpectrum(ft, fs))
    plt.show()


def Vowels(part):
    path = r"D:/Pavel/GitHub/PCInPhysics/python/sounds/"
    files = ["a.wav", "e.wav", "i.wav", "o.wav", "u.wav"]

    for file in files:
        sound, fs = Sound.Read(path + file)
        Sound.Play(sound, fs)

        sound = sound[part]

        N = len(sound)
        f = np.linspace(0, fs, N)

        ft1 = FourierTransform(sound)

        #plt.semilogy(f[0:1000], np.abs(ft)[0:1000], label=file)
        plt.plot(f, np.abs(ft1), label=file)


    plt.xlim(0,1000)
    plt.legend()
    plt.show()

def BlackHole():
    file = r"D:/Pavel/GitHub/PCInPhysics/python/sounds/BlackHolesCollision.wav"
    sound, fs = Sound.Read(file)
    #Sound.Play(sound, fs)

    sound = sound[:,0]

    N = len(sound)
    window = 1000
    
    i = 0
    Z = []
    Y = []
    X = []
    while i + window < N:
        ft = np.fft.fft(sound[i:(i + window)])
        f, a = AmplitudeSpectrum(ft, fs)
        Z.append(a)
        Y.append(f)
        X.append(np.linspace(i, i, window // 2))
        i += window

    plt.contourf(np.array(X), np.array(Y), np.array(Z))
    plt.ylim(0,500)
    plt.show()

#TestSignal()
#Vowels(slice(3000,5000))
BlackHole()