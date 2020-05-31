import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd

import FourierTransform

interval = 100  # Milliseconds
fs = 44100      # Standard sampling frequency

sound = sd.rec(int(interval * fs / 1000), samplerate=fs, channels=1)
sd.wait()                                       # Recording is asynchronous - we wait till it's finished

fig = plt.figure()
ax = plt.axes(xlim=(0, 10000), ylim=(0, 1.2))
                                           
line, = ax.plot([], [], lw=1, color="blue")     # Spectrogram
maximum, = ax.plot([], [], 'o', color="red", markersize=20)     # Maximal frequency

timeText = ax.text(100, 1.15, '')

def animate(i):
    global sound
    sd.wait()

    sound = sound[100:]                 # Getting rid of the first data of the time series (sometimes they contain some spurious noise)

    ft = np.fft.fft(sound[:,0])           # Fourier transform

    sound = sd.rec(int(interval * fs / 1000), samplerate=fs, channels=1)    # We start immediately a new measurements (to avoid time delays)

    frequencies, amplitudes = FourierTransform.AmplitudeSpectrum(ft, fs)

    maxIndex = np.argmax(amplitudes)           # Index of the maximum amplitude
    amplitudes /= amplitudes[maxIndex]         # Normalization

    line.set_data(frequencies, amplitudes)

    maximum.set_data(frequencies[maxIndex], amplitudes[maxIndex])    
    timeText.set_text(f"Dominant frequency = {frequencies[maxIndex]:3.0f} Hz")

    return line, maximum, timeText                        

anim = FuncAnimation(fig, animate, interval=interval, blit=True)
plt.show()

