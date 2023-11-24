import h5py
import matplotlib.pyplot as plt
from fluidsimulation import *

# How many timesteps in the simulation
timesteps = 1000

# Define the simulation
# n - size of the simulation
# windSpeed - speed of the wind
# windDirection - direction of the wind (in degrees)
# gasLocationX/Y - location of the source of the leak
# gasRelease - how much "particles" are released at the source  源强
# diffusion - diffusion of the gas  扩散系数
# viscosity - viscosity of the gas  粘度
# windNoise - how much the wind speed direction is changed as the simulation is unrolled (in degrees)  风向扰动噪声
# windNoiseTimestep - how often the wind speed direction is changed as the simulation is unrolled      风向扰动时间间隔
fs = FluidSimulation(64, windSpeed=0.1,
                     windDirection=60.0,
                     gasLocationX=32,
                     gasLocationY=32,
                     gasRelease=300,
                     diffusion=0.0001,
                     viscosity=0,
                     windNoise=90,
                     windNoiseTimestep=30)


# Save the data to a file
f = h5py.File("test.hdf5", "w")


# Run the simulation
for i in range(timesteps):
    fs.update(f)
    print("Sim: "+str(i))

# Load the data from the saved file
file = h5py.File("test.hdf5", "r")

# Visualize the data - every 10th image is shown
fig, ax = plt.subplots()
plt.ion()
for i in range(0, timesteps, 10):
    # Show the gas concentration as an image
    sc = ax.imshow(file['readings'+str(i)], vmin=0, vmax=10)
    if i == 0:
        cbar = plt.colorbar(sc, ax=ax)
    else:
        pass
    fig.savefig("./snapshots/"+str(i)+".png", dpi=300)
    plt.pause(0.2)
    ax.cla()
    print("Render: "+str(i))
plt.close(fig)
