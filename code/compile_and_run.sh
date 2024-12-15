#!/bin/bash

# Line to run: copy the two lines below in the prompt
# chmod +x compile_and_run.sh
# ./compile_and_run.sh

# Compile the program
gcc-10 -std=c++17 -fopenmp $(pkg-config --cflags --libs opencv4) -o LBM_esec LBM_2.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lstdc++ -lm


# Run the program
./LBM_esec

# Generate video from frames
if [ -d "frames" ]; then
  ffmpeg -framerate 10 -i frames/frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p simulation.mp4
fi
