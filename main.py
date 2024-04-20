import cv2
import h5py
import numpy as np
from fluidsimulation import FluidSimulation

# How many timesteps in the simulation
timesteps = 2000

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
cv2.namedWindow("Gas", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gas", 512, 512)
for i in range(0, timesteps, 10):
    img = np.array(file[f'frame{i}'])
    R = (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)
    G = np.zeros_like(R, dtype=np.uint8)
    B = np.zeros_like(R, dtype=np.uint8)
    BGR = cv2.merge([B, G, R])
    cv2.putText(BGR, f"{i//fs.FPS} s", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, 0, False)
    cv2.imwrite(f"./snapshots/{i:03d}.png", BGR)
    cv2.imshow("Gas", BGR)
    key = 0xFF & cv2.waitKey(200)
    if key == ord('q'):
        break

cv2.destroyWindow("Gas")
print("All Job Were Done!")
