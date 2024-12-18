#!/bin/bash

# Set executable permissions: chmod +x compile_and_run.sh
# Run the script: ./compile_and_run.sh [number_of_cores]

# Check for invalid input
if [ "$#" -gt 1 ]; then
  echo "Usage: ./compile_and_run.sh [number_of_cores]"
  exit 1
fi

# Set the number of cores
if [ "$#" -eq 1 ]; then
  NUM_CORES=$1
else
  NUM_CORES=$(nproc)  # Detects the number of CPU cores on the machine
fi

# Print the number of cores being used
echo "Running with $NUM_CORES cores"

# Compile the program
echo "Compiling the program..."
g++-10 -O3 -Wall -Wextra -march=native -std=c++17 -fopenmp $(pkg-config --cflags --libs opencv4) -o simulation_exec main.cpp LBM.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lstdc++ -lm

if [ $? -ne 0 ]; then
  echo "Compilation failed. Exiting."
  exit 1
fi

# Run the program
echo "Running the simulation..."
./simulation_exec $NUM_CORES

# Generate video from frames
if [ -d "frames" ]; then
  echo "Generating video from frames..."
  ffmpeg -framerate 10 -i frames/frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p simulation.mp4
  echo "Video saved as simulation.mp4"
else
  echo "No frames directory found. Skipping video generation."
fi
