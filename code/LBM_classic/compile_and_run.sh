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
g++-10 -O3 -Wall -Wextra -march=native -std=c++20 -fopenmp \
$(pkg-config --cflags --libs opencv4) -o simulation_exec \
main.cpp LBM.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lm

if [ $? -ne 0 ]; then
  echo "Compilation failed. Exiting."
  exit 1
fi

# Run the program
echo "Running the simulation..."
./simulation_exec $NUM_CORES
