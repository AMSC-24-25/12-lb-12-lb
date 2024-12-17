#!/bin/bash

#LINES to print #substitue 8 with the desired number of cores or leave blank if you want to run with the maximum number of cores
#chmod +x compile_and_run.sh
#./compile_and_run.sh 8

#Check for mistake in the input
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
gcc-10 -std=c++17 -fopenmp $(pkg-config --cflags --libs opencv4) -o LBM_esec LBM_3.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lstdc++ -lm

# Run the program
./LBM_esec $NUM_CORES

# Generate video from frames
if [ -d "frames" ]; then
  ffmpeg -framerate 10 -i frames/frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p simulation.mp4
fi
