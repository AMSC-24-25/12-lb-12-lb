#need to install ffmpeg
#sudo apt-get install ffmpeg


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


# It will find in the directory all files named with the same pattern
file_list = sorted(glob.glob("output_*.csv")) 
num_files = len(file_list)

if num_files == 0:
    print("No file found")
else:
    # Create a directory for plots
    os.makedirs("Velocity_plots",exist_ok=True)
    os.makedirs("Density_plots", exist_ok=True)

    def read_data(file):
        df=pd.read_csv(file)
        x=df['x']
        y = df['y']
        u_x = df['u_x']
        u_y = df['u_y']
        rho = df['rho']
        return x, y, u_x, u_y, rho

    def plot_velocity(x,y,u_x, u_y,t):
        plt.figure(figsize=(6, 6))
        plt.quiver(x, y, u_x, u_y, scale=50)
        plt.title(f"Velocit\u00e0 a step={t}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(0, max(x))
        plt.ylim(0, max(y))
        plt.grid()
        plt.savefig(f"Velocity_plots/Velocity_{t}.png", dpi=300)
        plt.close()
    
    def plot_density(x, y, rho, t):
        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, c=rho, cmap='viridis')
        plt.colorbar(label='Densit\u00e0')
        plt.title(f"Densit\u00e0 a step={t}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(0, max(x))
        plt.ylim(0, max(y))
        plt.grid()
        plt.savefig(f"Density_plots/Density_{t}.png", dpi=300)
        plt.close()
    
    #Static plots generation
    for i, file in enumerate(file_list):
        x, y, u_x, u_y, rho = read_data(file)
        plot_velocity(x, y, u_x, u_y, i)
        plot_density(x, y, rho, i)
    
    def animate_velocity(i):
        img = plt.imread(f"Velocity_plots/Velocity_{i}.png")
        ax.clear()
        ax.imshow(img)
        ax.axis('off')
    def animate_density(i):
        img = plt.imread(f"Density_plots/Density_{i}.png")
        ax.clear()
        ax.imshow(img)
        ax.axis('off')

    # Velocity Animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ani_velocity = FuncAnimation(fig, animate_velocity, frames=num_files, interval=100)
    ani_velocity.save('Velocity_animation.mp4', writer='ffmpeg', fps=10)

    # Density Animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ani_density = FuncAnimation(fig, animate_density, frames=num_files, interval=100)
    ani_density.save('Density_animation.mp4', writer='ffmpeg', fps=10)

    print("Grafici e animazioni salvati.")
  
