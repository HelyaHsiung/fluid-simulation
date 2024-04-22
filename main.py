import cv2
import h5py
import numpy as np
from fluidsimulation import FluidSimulation

# How many timesteps in the simulation
timesteps = 10000

scale = 1/2
fs = FluidSimulation(
    width=int(640*scale),
    height=int(480*scale),
    gasRelease=55,
    gasLocationX=int(260*scale),
    gasLocationY=int(220*scale),
    simu_windDirection=135,
    simu_windSpeed=0.04,
    simu_wind_step=20,
    simu_windDirectionNoise_range=180,  # degree
    simu_windSpeedNoise_range=0.4,     # percent
    )

# Save the data to a file
f = h5py.File("test.hdf5", "w")

# Run the simulation
cv2.namedWindow("Running", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Running", 640, 480)
for i in range(timesteps):
    fs.update(f)
    img_r = np.array(fs.d.reshape((fs.h + 2, fs.w + 2))[1:fs.h + 1, 1:fs.w + 1])
    R_r = (255 * (img_r - img_r.min()) / (img_r.max() - img_r.min())).astype(np.uint8)
    G_r = np.zeros_like(R_r, dtype=np.uint8)
    B_r = np.zeros_like(R_r, dtype=np.uint8)
    BGR_r = cv2.merge([B_r, G_r, R_r])
    cv2.imshow("Running", BGR_r)
    print(f"Sim: {i}")
    key_r = 0xFF & cv2.waitKey(1)
    if key_r == ord('q'):
        break
cv2.destroyWindow("Running")
print("Simulation Done!")

# Load the data from the saved file
file = h5py.File("test.hdf5", "r")

# Visualize the data - every 10th image is shown
cv2.namedWindow("Gas", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gas", 640, 480)
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
