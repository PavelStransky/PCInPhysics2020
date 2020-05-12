import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm           # Colour maps for the contour graph

import time

import Sound
from FourierTransform import *


def TestSignal(N=2000, fs=2000):
    """ Task 9.3 """
    x = np.linspace(0, N / fs, N, endpoint=False)
    y = 0.1 * np.sin(440 * 2 * np.pi * x) + 0.2 * np.sin(5 / 4 * 440 * 2 * np.pi * x) + 0.3 * np.sin(3 / 2 * 440 * 2 * np.pi * x)
    Sound.Play(y, fs)

    ft = FourierTransform(y)
    plt.plot(*AmplitudeSpectrum(ft, fs))
    plt.title(f"$f_s={fs}$ Hz")
    plt.xlabel("$f$ [Hz]")
    plt.ylabel("$A$")
    plt.show()


def Vowels(part):
    """ Task 9.4 """
    path = r"sounds/"
    files = ["a.wav", "e.wav", "i.wav", "o.wav", "u.wav"]

    for file in files:
        sound, fs = Sound.Read(path + file)
        Sound.Play(sound, fs)

        sound = sound[part]

        ft = FourierTransform(sound)
        plt.plot(*AmplitudeSpectrum(ft, fs), label=file)

    plt.xlim(0, 2000)       # Just the lowest part of the spectrum (it contains the important frequencies)
    plt.xlabel("$f$ [Hz]")
    plt.ylabel("$A$")
    plt.legend()
    plt.show()


def FTMethodComparison():
    """ Task 9.6 """
    file = r"sounds/a.wav"

    sound, fs = Sound.Read(file)
    Sound.Play(sound, fs)

    sound = sound[3000:5000]
    N = len(sound)

    ft1 = FourierTransform(sound)
    plt.plot(*AmplitudeSpectrum(ft1, fs), label="FourierTransform")

    ft2 = np.fft.fft(sound) / N
    plt.plot(*AmplitudeSpectrum(ft2, fs), label="numpy.fft.fft")

    plt.xlabel("$f$ [Hz]")
    plt.ylabel("$A$")
    plt.legend()
    plt.show()


def BlackHoles(windowSize=2000, step=100):
    """ Task 9.5 """
    file = r"sounds/BlackHolesCollision.wav"
    sound, fs = Sound.Read(file)
    Sound.Play(sound, fs)

    sound = sound[:, 0]      # The input signal has two channels (amplitude and time) - we'll need just the amplitude
    N = len(sound)
    
    X, Y, Z = [], [], []    # For the contour plot 

    i = 0                   # Index of the window beginning

    while i + windowSize < N:
        window = sound[i:(i + windowSize)]
        
        ft = np.fft.fft(window) / windowSize
        #ft = FourierTransform(window)
        
        frequencies, amplitudes = AmplitudeSpectrum(ft, fs)
        X.append(np.linspace(i / fs, i / fs, windowSize // 2))
        Y.append(frequencies)
        Z.append(amplitudes)
        i += step

    plt.contourf(np.array(X), np.array(Y), np.array(Z), cmap=cm.hot)
    plt.xlabel("$t$ [s]")
    plt.ylabel("$f$ [Hz]")
    plt.colorbar()
    plt.ylim(0, 500)
    plt.show()

#TestSignal(2000, 2000)
#TestSignal(1000, 1000)
#Vowels(slice(3000, 5000))
BlackHoles()
FTMethodComparison()
