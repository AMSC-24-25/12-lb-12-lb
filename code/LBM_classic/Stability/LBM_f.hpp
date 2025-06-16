#ifndef LBM_H
#define LBM_H

#include <vector>
#include <array>
#include <utility>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>        
#include <opencv2/imgproc.hpp>   
#include <opencv2/highgui.hpp> 
#include <opencv2/videoio.hpp> 
#include <filesystem>


class LBmethod {
private:
    // Parameters
    const size_t NSTEPS;
    const double u_lid;
    const double Re;
    double u_lid_dyn;
    const size_t num_cores;

    // Fixed parameters
    const size_t ndirections = 9;
    double nu; 
    double tau;
    const double c_s; //Characteristic lattice speed D2Q9 model
    const double sigma = 10.0;

    // Directions and weights for D2Q9
    const std::array<int, 9> directionx;
    const std::array<int, 9> directiony;
    const std::array<double, 9> weight;

    // Simulation data
    std::vector<double> rho;
    std::vector<double> ux;
    std::vector<double> uy;
    std::vector<double> f_eq;
    std::vector<double> f;
    std::vector<double> f_temp;

    // Overloaded function for 2D to 1D indexing
    inline size_t INDEX(size_t x, size_t y, size_t NX) {
        return x + NX * y;
    }

    // Overloaded function for 3D to 1D indexing
    inline size_t INDEX(size_t x, size_t y, size_t i, size_t NX, size_t ndirections) {
        return i + ndirections * (x + NX * y);
    }

    cv::VideoWriter video_writer;

public:
    size_t NX;
    size_t NY;
    // Constructor
    LBmethod(const size_t NSTEPS, size_t NX, size_t NY, const double u_lid, const double Re, const size_t num_cores);

    // Methods
    void Initialize();
    void Equilibrium();
    void UpdateMacro();
    void Collisions();
    void Streaming();
    void Run_simulation();
    void Visualization(size_t t);
};

#endif // LBMETHOD_H
