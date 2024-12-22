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
    const size_t NX;
    const size_t NY;
    const double Re;
    const size_t num_cores;

    // Fixed parameters
    const size_t ndirections = 9;
    const double tau_ion;
    const double tau_el;
    const double q_ion=1.6 * 1e-19;
    const double q_el= - 1.6 * 1e-19;;
    const double sigma = 10.0;

    // Directions and weights for D2Q9
    const std::array<int, 9> directionx;
    const std::array<int, 9> directiony;
    const std::array<double, 9> weight;

    // Simulation data
    std::vector<double> rho_ion;
    std::vector<double> rho_el;

    std::vector<double> ux_ion;
    std::vector<double> ux_el;

    std::vector<double> uy_ion;
    std::vector<double> uy_el;

    std::vector<double> f_eq_ion;
    std::vector<double> f_eq_el;

    std::vector<double> f_ion;
    std::vector<double> f_el;

    std::vector<double> f_temp_ion;
    std::vector<double> f_temp_el;

    std::vector<double> phi;
    std::vector<double> phi_new;

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
    // Constructor
    LBmethod(const size_t NSTEPS, const size_t NX, const size_t NY, const double Re, const size_t num_cores, const size_t tau_ion, const size_t tau_el);

    // Methods
    void Initialize();
    void Equilibrium();
    void UpdateMacro();
    void SolvePoisson();
    void Collisions();
    void Streaming();
    void Run_simulation();
    void Visualization(size_t t);
};

#endif // LBMETHOD_H

