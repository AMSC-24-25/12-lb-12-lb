Steps of the code implementation
1) write a code that actually work for the study case
2) implement class and other stuff from the AMSC course
3) optimize
4) parallelize
5) optimize even more
6) (optional) extend to 3D domain

---------------------------------------------------------------------------------------------------------------------------
gcc-10 -std=c++17 -fopenmp $(pkg-config --cflags --libs opencv4) -o LBM2_eseg LBM_2.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lstdc++ -lm

./LBM2_eseg
