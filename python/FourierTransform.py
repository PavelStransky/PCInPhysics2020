import numpy as np

def FourierTransform(signal):
    """ Discrete Fourier Transform """
    N = len(signal)

    argument = -2j * np.pi / N * np.arange(N)
    """ Temporary array with exponents for the exponential 
        - uses more memory, but makes the calculation much faster
    """

    H = np.array([sum(signal * np.exp(k * argument)) for k in range(N)]) / N
    return H


def InverseFourierTransform(signal):
    """ Inverse Discrete Fourier Transform """
    N = len(signal)

    argument = 2j * np.pi / N * np.arange(N)
    H = np.array([sum(signal * np.exp(k * argument)) for k in range(N)])
    return H


def AmplitudeSpectrum(components, fs):
    """ Spectrum of Fourier amplitudes.

        Arguments:
        fs - Sampling frequency
    """
    N = len(components)
    powerSpectrum = 2 * np.abs(components)
    powerSpectrum[0] = 0.5 * powerSpectrum[0]       # DC term is there just once
    return np.linspace(0, fs // 2, N // 2), powerSpectrum[0:(N // 2)]