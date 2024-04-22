import cv2
import h5py
import numpy as np
from fluidsimulation import FluidSimulation


def gen_video(timesteps):
    scale = 1/4
    fs = FluidSimulation(
        width=int(640*scale),
        height=int(480*scale),
        gasRelease=100,
        gasLocationX=int(260*scale),
        gasLocationY=int(220*scale),
        simu_windDirection=170,
        simu_windSpeed=0.01,
        simu_wind_step=100,
        simu_windDirectionNoise_range=180,  # degree
        simu_windSpeedNoise_range=0.5,     # percent
        )

    # Save the data to a file
    f = h5py.File("test.hdf5", "w")

    # Run the simulation
    cv2.namedWindow("Running", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Running", 640, 480)
    for i in range(timesteps):
        fs.update(f)
        
        img = np.array(fs.d.reshape((fs.h + 2, fs.w + 2))[1:fs.h + 1, 1:fs.w + 1])
        
        R = (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)
        G = np.zeros_like(R, dtype=np.uint8)
        B = np.zeros_like(R, dtype=np.uint8)
        
        BGR = cv2.merge([B, G, R])
        
        cv2.putText(BGR, f"{i//fs.FPS}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1, 0, False)
        cv2.imshow("Running", BGR)
        cv2.imwrite(f"./snapshots/{i:03d}.png", R)
        
        print(f"Simulation: {i:03d}")
        
        key_r = 0xFF & cv2.waitKey(1)
        if key_r == ord('q'):
            break
    cv2.destroyWindow("Running")


def display_video(timesteps):
    file = h5py.File("test.hdf5", "r")
    cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
    for i in range(0, timesteps, 10):
        img = file[f"frame{i:03d}"][:]
        
        R = (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)
        G = np.zeros_like(R, dtype=np.uint8)
        B = np.zeros_like(R, dtype=np.uint8)
        
        BGR = cv2.merge([B, G, R])
        
        cv2.putText(BGR, f"{i//30}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1, 0, False)
        cv2.imshow("Display", BGR)
        
        key_r = 0xFF & cv2.waitKey(1)
        if key_r == ord('q'):
            break
    cv2.destroyWindow("Display")


if __name__ == "__main__":
    # How many timesteps in the simulation
    timesteps = 10000
    gen_video(timesteps)
    display_video(timesteps)
    print("All Job Were Done!")
