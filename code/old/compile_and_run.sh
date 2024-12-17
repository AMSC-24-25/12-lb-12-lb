#!/bin/bash

# Usage:
# ./compile_and_run.sh <version> [number_of_threads]
# Where:
#   <version> is 1, 2, or 3
#   [number_of_threads] is optional and only allowed for version 3

# Check for valid input
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: ./compile_and_run.sh <version> [number_of_threads]"
  echo "       <version> can be 1, 2, or 3"
  echo "       [number_of_threads] is only allowed for version 3"
  exit 1
fi

# Check if the first argument is valid (1, 2, or 3)
if [[ "$1" != "1" && "$1" != "2" && "$1" != "3" ]]; then
  echo "Error: Invalid version '$1'. Valid options are 1, 2, or 3."
  exit 1
fi

# Assign version and thread count
LBM_VERSION="$1"

# Handle thread count logic for LBM_3
if [ "$LBM_VERSION" == "3" ]; then
  if [ "$#" -eq 2 ]; then
    NUM_CORES="$2"
  else
    NUM_CORES=$(nproc) # Auto-detect number of cores if not specified
  fi
  echo "Running LBM_$LBM_VERSION with $NUM_CORES threads"
elif [ "$#" -eq 2 ]; then
  # For LBM_1 or LBM_2, thread count is not allowed
  echo "Error: Thread count is not allowed for LBM_$LBM_VERSION."
  exit 1
else
  echo "Running LBM_$LBM_VERSION"
fi

# Compile the appropriate file
echo "Compiling LBM_$LBM_VERSION..."
gcc-10 -std=c++17 -fopenmp $(pkg-config --cflags --libs opencv4) -o LBM_esec LBM_$LBM_VERSION.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lstdc++ -lm
if [ $? -ne 0 ]; then
  echo "Compilation failed."
  exit 1
fi

# Run the program
if [ "$LBM_VERSION" == "3" ]; then
  ./LBM_esec $NUM_CORES
else
  ./LBM_esec
fi

# Generate video from frames
if [ -d "frames" ]; then
  echo "Generating video from frames..."
  ffmpeg -framerate 10 -i frames/frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p simulation.mp4
fi
