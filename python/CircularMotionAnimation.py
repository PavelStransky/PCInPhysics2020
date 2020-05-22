import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Model parameters
radius = 1
period = 10             # In seconds
interval = 50           # In milliseconds

axesLimits = (-1.1 * radius, 1.1 * radius)

fig = plt.figure()                                              # Create the figure object (the window)
ax = plt.axes(xlim=axesLimits, ylim=axesLimits, aspect="equal") # Create the axes object (the frame in the figure with the graph); 
                                                                # aspect="equal" guaranties that the circle will not be deformed
line, = ax.plot([], [], 'o-', lw=2, color="red")                # Create an empty line in the frame

timeText = ax.text(-1, 1, '')                                   # Text with actual time at given coordinates

def animate(i):
    """ Main function - plots the i-th frame """
    t = i * interval / 1000
    phase = 2 * np.pi * t / period
    x = np.array([0, np.sin(phase)])
    y = np.array([0, np.cos(phase)])
    line.set_data(x, y)

    timeText.set_text(f"Time = {t:0.2f}")

    return line, timeText                        # It is necessary to return all modified objects (if blit=True), otherwise it can be ommitted.

anim = FuncAnimation(fig, animate, interval=interval, blit=True)

#anim = FuncAnimation(fig, animate, frames=200, interval=interval, blit=True)
#anim.save("circular.gif", writer='imagemagick')
#anim.save("circular.mp4", extra_args=['-vcodec', 'libx264'])

plt.show()
