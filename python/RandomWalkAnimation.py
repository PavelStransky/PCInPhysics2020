import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import mpl_toolkits.mplot3d.axes3d as axes3d

import RandomWalk

interval = 50           # In milliseconds
boxSize = 20

randomWalk3D = RandomWalk.RandomWalk(dimension=3, boxSize=boxSize)

axesLimits = (-boxSize, boxSize)

fig = plt.figure()
ax = axes3d.Axes3D(fig, xlim=axesLimits, ylim=axesLimits, zlim=axesLimits)
line, = ax.plot([], [], [])

def animate(i):
    line.set_data(randomWalk3D[0:i, 0], randomWalk3D[0:i, 1])   # set_data accepts x and y coordinates only
    line.set_3d_properties(randomWalk3D[0:i, 2])                # z coordinates must be given by this function
    ax.view_init(30, i)                                         # In order to make it even fancier, we rotate the graph at each step
    return line, ax

anim = FuncAnimation(fig, animate, interval=interval, blit=True, frames=len(randomWalk3D))

# If we want to save the animation, we have to switch off blit!!!
#anim = FuncAnimation(fig, animate, interval=interval, frames=len(randomWalk3D))
#anim.save("randomwalk.gif", writer='imagemagick')
#anim.save("randomwalk.mp4", extra_args=['-vcodec', 'libx264'])

plt.show()