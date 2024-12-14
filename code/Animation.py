import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from concurrent.futures import ProcessPoolExecutor

# Get the list of CSV files
file_list = sorted(glob.glob("output_*.csv"))
num_files = len(file_list)

if num_files == 0:
    print("No file found")
else:
    animation_output_file = "Flow_Visualization_Animation.mp4"

    def read_data(file):
        # Read CSV file and return reshaped fields
        df = pd.read_csv(file)
        x = df['x'].values
        y = df['y'].values
        u_x = df['u_x'].values
        u_y = df['u_y'].values
        rho = df['rho'].values
        # Create a uniform grid from unique x and y values
        x_unique = np.unique(x)
        y_unique = np.unique(y)
        x, y = np.meshgrid(x_unique, y_unique)  # Generate grid
        Nx, Ny = x.shape

        # Reshape u_x, u_y, and rho to match the grid
        u_x = u_x.reshape((Ny, Nx))  # Note the order Ny, Nx
        u_y = u_y.reshape((Ny, Nx))
        rho = rho.reshape((Ny, Nx))
        return x, y, u_x, u_y, rho

    # Read the first file to set up the grid
    x, y, u_x, u_y, rho = read_data(file_list[0])
    u_magn = np.sqrt(u_x**2 + u_y**2)  # Velocity magnitude

    # Create a figure for plotting, 2 subplots
    fig, axs = plt.subplots(1,2,figsize=(14, 6))

    # Velocity magnitude background (heatmap)+ streamlines
    ax1=axs[0]
    velocity_background = ax1.imshow(
        u_magn.T, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()], cmap="plasma"
    )
    
    stream = ax1.streamplot(x, y, u_x, u_y, color=u_magn, linewidth=1, cmap="viridis")
    ax1.set_title("Velocity Field")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    cbar1 = fig.colorbar(velocity_background, ax=ax1, orientation="vertical", label="Velocity Magnitude")

    #subplot 2: Density field
    ax2 = axs[1]
    density_background = ax2.imshow(
        rho.T, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()], cmap="cividis"
    )
    ax2.set_title("Density Field")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    cbar2 = fig.colorbar(density_background, ax=ax2, orientation="vertical", label="Density")

    def update(frame):
        # Read the data for the current frame
        x, y, u_x, u_y, rho = read_data(file_list[frame])
        u_magn = np.sqrt(u_x**2 + u_y**2)

        velocity_background.set_data(u_magn.T)
        density_background.set_data(rho.T)

        return velocity_background, density_background
    
    def preload_data():
        with ProcessPoolExecutor() as executor:
            results= list(executor.map(read_data, file_list))
        return results
    print("Preloading data in parallel...")
    all_frames_data=preload_data()
    
    def update_preload(frame):
        x, y, u_x, u_y, rho = all_frames_data[frame]
        u_magn = np.sqrt(u_x**2 + u_y**2)

        velocity_background.set_data(u_magn.T)
        density_background.set_data(rho.T)

        return velocity_background, density_background
    
    # Create the animation (blit does not work with streamlines)
    ani = FuncAnimation(fig, update_preload, frames=num_files, interval=200, blit=True)

    # Save the animation as a video
    writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(animation_output_file, writer=writer)

    print(f"Animation saved as: {animation_output_file}")
