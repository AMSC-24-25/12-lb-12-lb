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

    weak_df = config_df.copy()
    weak_df["load_per_core"] = (weak_df["Grid_Size"] / weak_df["Number_of_Cores"]).round().astype(int)

    unique_loads = sorted(weak_df["load_per_core"].unique())

    # === Weak scalability, all loads ===
    # Loads with tollerance 5%
    target_loads = [1000, 2500, 5000]  # It can be changed
    tolerance = 0.05

    for load in target_loads:
        low = load * (1 - tolerance)
        high = load * (1 + tolerance)
        data = weak_df[(weak_df["load_per_core"] >= low) & (weak_df["load_per_core"] <= high)]

        if len(data["Number_of_Cores"].unique()) > 1:
            plt.figure()
            data = data.sort_values("Number_of_Cores")
            plt.errorbar(data["Number_of_Cores"], data["Total_Computation_Time(ms)"],
                         yerr=data["std"], fmt='o-', capsize=5)
            plt.xlabel("Number of Cores")
            plt.ylabel("Computation Time (ms)")
            plt.title(f"Weak scalability - Load/Core ≈ {load}\nBC: {label_bc}, Poisson: {label_poisson}")
            plt.grid(True)
            filename = f"{output_dir}/Weak scalability_{label_bc}_{label_poisson}_LoadApprox{load}.png"
            plt.savefig(filename)
            plt.close()

    # === Weak scalability -comparison ===
    plt.figure(figsize=(10, 6))
    for load in target_loads:
        low = load * (1 - tolerance)
        high = load * (1 + tolerance)
        data = weak_df[(weak_df["load_per_core"] >= low) & (weak_df["load_per_core"] <= high)]

        if len(data["Number_of_Cores"].unique()) > 1:
            data = data.sort_values("Number_of_Cores")
            plt.errorbar(data["Number_of_Cores"], data["Total_Computation_Time(ms)"],
                        yerr=data["std"], label=f"Load/Core ≈ {load}", capsize=3, fmt='o-')

    plt.xlabel("Number of Cores")
    plt.ylabel("Computation Time (ms)")
    plt.title(f"Weak scalability comparison\nBC: {label_bc}, Poisson: {label_poisson}")
    plt.legend()
    plt.grid(True)
    filename = f"{output_dir}/Weak scalability COMPARISON_{label_bc}_{label_poisson}.png"
    plt.savefig(filename)
    plt.close()
