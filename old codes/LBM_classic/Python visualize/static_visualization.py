import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Upload data from CSV file 
data = pd.read_csv("output_equilibrium.csv")

# Domain dimensions
NX = data['x'].max() + 1
NY = data['y'].max() + 1

# Reshape
rho = data.pivot(index='y', columns='x', values='rho').values

# Density map
plt.figure(figsize=(8, 6))
plt.imshow(rho, origin='lower', cmap='viridis', interpolation='nearest')
plt.colorbar(label='Density')
plt.title('Equilibrium Density Map')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("grafico.png")  # Save as PNG file
