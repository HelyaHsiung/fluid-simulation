import h5py
import matplotlib.pyplot as plt
from fluidsimulation_2 import *

# How many timesteps in the simulation
plt.ion()
timesteps = 500

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

fs = FluidSimulation()

# Save the data to a file
f = h5py.File("test.hdf5", "w")

# Run the simulation
for i in range(timesteps):
    fs.update(f)
    print("Sim: "+str(i))
print("Simulation Done!")

# Load the data from the saved file
file = h5py.File("test.hdf5", "r")

# Visualize the data - every 10th image is shown
fig, ax = plt.subplots()
plt.ion()
for i in range(0, timesteps, 10):
    # Show the gas concentration as an image
    sc = ax.imshow(file['frame'+str(i)], vmin=0, vmax=fs.gasRelease, cmap='Reds')
    if i == 0:
        cbar = plt.colorbar(sc, ax=ax)
    else:
        pass
    fig.savefig("./snapshots/"+str(i)+".png", dpi=300)
    plt.pause(0.0001)
    ax.cla()
plt.close(fig)

print("All Job Were Done!")
