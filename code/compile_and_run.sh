#!/bin/bash

# Set environment variables
#export LD_LIBRARY_PATH=/u/sw/toolchains/gcc-glibc/11.2.0/prefix/lib:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Compile the program
gcc-10 -std=c++17 -fopenmp $(pkg-config --cflags --libs opencv4) -o LBM_esec LBM_2.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lstdc++ -lm


# Run the program
./LBM_esec

# Generate video from frames
if [ -d "frames" ]; then
  ffmpeg -framerate 10 -i frames/frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p simulation.mp4
fi
