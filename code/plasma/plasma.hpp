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
    const size_t num_cores;

    // Fixed parameters
    const size_t ndirections = 9;
    const size_t Z_ion;
    const size_t A_ion;
    const double r_ion;
    const double tau_ion;
    const double tau_el;
    const double q_el= - 1.6021766 * 1e-19; //C
    const double kb=1.380649*1e-23; //J/K
    const double m_el=-9.010938356*1e-31; //Kg
    const double m_p=1.672621922369*1e-27;//Kg
    const double eps_0=8.8541878128*1e-12;//F/m
    const double r_el=2.8179*1e-15;//m
    const double Bz_ext = 0.0;  // magnitude of magnetic field in z-direction
    const double Ex_ext = 100.0;  // magnitude of electric field in x-direction
    const double Ey_ext = 0.0;  // magnitude of electric field in x-direction

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

    //Function to reduce the lenght of program
    inline void apply_colormap(const cv::Mat& input_32f, cv::Mat& output_8uc3, int colormap) {
        static cv::Mat tmp_8u;
        cv::normalize(input_32f, tmp_8u, 0, 255, cv::NORM_MINMAX);
        tmp_8u.convertTo(tmp_8u, CV_8U);
        cv::applyColorMap(tmp_8u, output_8uc3, colormap);
    }

    cv::VideoWriter video_writer;

public:
    // Constructor
    LBmethod(const size_t NSTEPS, const size_t NX, const size_t NY, const size_t num_cores, const size_t Z_ion, const size_t A_ion, const double r_ion, const double tau_ion, const double tau_el);

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
