import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata

# Mappature
bc_map = {0: "Periodic", 1: "BounceBack"}
poisson_map = {0: "NONE", 1: "GS", 2: "SOR", 3: "FFT", 4: "9point"}

# === Lettura CSV ===
df = pd.read_csv("simulation_time_plasma_details.csv")

# Estrazione NX, NY, Grid_Size da stringa "NXxNY"
def parse_grid(grid_str):
    match = re.match(r"(\d+)x(\d+)", grid_str)
    if match:
        nx, ny = map(int, match.groups())
        return nx, ny, nx * ny
    return 0, 0, 0

df[["NX", "NY", "Grid_Size"]] = df["Grid_Dimension"].apply(lambda s: pd.Series(parse_grid(s)))

# === Calcolo media e std ===
grouped = df.groupby(["Grid_Dimension", "Grid_Size", "Number_of_Steps", "Number_of_Cores", "Poisson", "BC"])
mean_df = grouped["Total_Computation_Time(ms)"].mean().reset_index()
std_df = grouped["Total_Computation_Time(ms)"].std().reset_index()
mean_df["std"] = std_df["Total_Computation_Time(ms)"]

# === Cartella output ===
output_dir = "Scalability tests"
os.makedirs(output_dir, exist_ok=True)

# === Ciclo per configurazioni distinte ===
configs = mean_df[["Poisson", "BC"]].drop_duplicates()

for _, config in configs.iterrows():
    poisson = config["Poisson"]
    bc = config["BC"]
    label_poisson = poisson_map[poisson]
    label_bc = bc_map[bc]

    config_df = mean_df[(mean_df["Poisson"] == poisson) & (mean_df["BC"] == bc)]

    # === Strong scalability ===
    for grid_label in config_df["Grid_Dimension"].unique():
        data = config_df[config_df["Grid_Dimension"] == grid_label]
        if len(data["Number_of_Cores"].unique()) > 1:
            plt.figure()
            data = data.sort_values("Number_of_Cores")
            plt.errorbar(data["Number_of_Cores"], data["Total_Computation_Time(ms)"],
                         yerr=data["std"], fmt='o-', capsize=5)
            plt.xlabel("Number of Cores")
            plt.ylabel("Computation Time (ms)")
            plt.title(f"Strong scalability - Grid {grid_label}\nBC: {label_bc}, Poisson: {label_poisson}")
            plt.grid(True)
            filename = f"{output_dir}/Strong scalability_{label_bc}_{label_poisson}_Grid{grid_label}.png"
            plt.savefig(filename)
            plt.close()


    # === Grid size impact ===
    for core in config_df["Number_of_Cores"].unique():
        data = config_df[config_df["Number_of_Cores"] == core]
        if len(data["Grid_Size"].unique()) > 1:
            plt.figure()
            data = data.sort_values("Grid_Size")
            plt.errorbar(data["Grid_Size"], data["Total_Computation_Time(ms)"],
                         yerr=data["std"], fmt='o-', capsize=5)
            plt.xlabel("Grid Size (NX * NY)")
            plt.ylabel("Computation Time (ms)")
            plt.title(f"Grid size impact - Cores={core}\nBC: {label_bc}, Poisson: {label_poisson}")
            plt.grid(True)
            filename = f"{output_dir}/Grid size impact_{label_bc}_{label_poisson}_Cores{core}.png"
            plt.savefig(filename)
            plt.close()



    # === 3D Scalability Surface Plot ===
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Coordinate originali (irregolari)
    X = config_df["Number_of_Cores"].to_numpy()
    Y = config_df["Grid_Size"].to_numpy()
    Z = config_df["Total_Computation_Time(ms)"].to_numpy()

    # Definiamo una griglia regolare
    xi = np.linspace(X.min(), X.max(), 30)
    yi = np.linspace(Y.min(), Y.max(), 30)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolazione sulla griglia regolare
    Zi = griddata((X, Y), Z, (Xi, Yi), method='linear')

    # Rappresentazione della superficie
    surf = ax.plot_surface(Xi, Yi, Zi, cmap='viridis', edgecolor='none', alpha=0.9)

    # Impostazioni del grafico
    ax.set_xlabel("Number of Cores")
    ax.set_ylabel("Grid Size (NX * NY)")
    ax.set_zlabel("Computation Time (ms)")
    ax.set_title(f"3D Scalability Surface\nBC: {label_bc}, Poisson: {label_poisson}")

    # Barra del colore
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Computation Time (ms)")

    # Salva immagine
    filename = f"{output_dir}/3D scalability surface_{label_bc}_{label_poisson}.png"
    plt.savefig(filename)
    plt.close()

