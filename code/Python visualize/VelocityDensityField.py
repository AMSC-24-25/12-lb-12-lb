import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Get the list of CSV files
file_list = sorted(glob.glob("output_*.csv"))
num_files = len(file_list)

if num_files == 0:
    print("No file found")
else:
    # Create output directories
    frame_output_dir = "Flow_Visualization_Frames"
    os.makedirs(frame_output_dir, exist_ok=True)
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

    # Function to update the plot for each frame
    def update(frame):
        # Read the data for the current frame
        x, y, u_x, u_y, rho = read_data(file_list[frame])
        u_magn = np.sqrt(u_x**2 + u_y**2)

        for collection in ax1.collections:
            collection.remove()

        # Update velocity magnitude background and streamlines
        velocity_background.set_data(u_magn.T)
        #ax1.collections[:]=[]  # Clear previous streamlines
        stream=ax1.streamplot(x, y, u_x, u_y, color=u_magn, linewidth=1, cmap="viridis")

        # Update density field
        density_background.set_data(rho.T)

        # Save the current frame as an image
        frame_file = os.path.join(frame_output_dir, f"frame_{frame:04d}.png")
        plt.savefig(frame_file, dpi=300)

    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_files, interval=200)

    # Save the animation as a video
    writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(animation_output_file, writer=writer)

    print(f"Frames saved in directory: {frame_output_dir}")
    print(f"Animation saved as: {animation_output_file}")


