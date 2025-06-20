#!/bin/bash

# Set executable permissions: chmod +x compile_and_run_plasma.sh
# Run the script: ./compile_and_run_plasma.sh [number_of_cores]

export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH

# Usage check
if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [number_of_cores]"
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

# Source files
SRCS=$(find src -name '*.cpp')

# Compiler and flags
CXX=g++-10
OPT_FLAGS="-O3 -Wall -Wextra -march=native -std=c++20 -fopenmp"
PKG_FLAGS="$(pkg-config --cflags --libs opencv4)"
INCLUDE_FLAGS="-Iinclude"
EXTRA_LIBS="-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lfftw3 -lm"

# Build
echo "Compiling the program..."
$CXX $OPT_FLAGS $INCLUDE_FLAGS $PKG_FLAGS $SRCS -o simulation_exec $EXTRA_LIBS


if [ $? -ne 0 ]; then
  echo "Compilation failed. Exiting."
  exit 1
fi

# Run
echo "Compilation successful. Launching simulation..."
./simulation_exec $NUM_CORES
