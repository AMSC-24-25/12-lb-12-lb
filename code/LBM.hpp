#ifndef LBM_H
#define LBM_H

#include <vector>
#include <array>
#include <utility>
#include <string>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

class LBmethod {
private:
    // Parameters
    const unsigned int NSTEPS;
    const unsigned int NX;
    const unsigned int NY;
    const double u_lid;
    const double Re;
    const double rho0;
    double u_lid_dyn;
    const unsigned int num_cores;

    // Fixed parameters
    const unsigned int ndirections = 9;
    const double nu;
    const double tau;
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

    // Temporary variables for calculations
    double rho_local;  // Temporary density accumulator
    double ux_local;   // Temporary x-velocity accumulator
    double uy_local;   // Temporary y-velocity accumulator
    double u2;
    double cu;

    // Overloaded function for 2D to 1D indexing
    inline int INDEX(int x, int y, int NX) {
        return x + NX * y;
    }

    // Overloaded function for 3D to 1D indexing
    inline int INDEX(int x, int y, int i, int NX, int ndirections) {
        return i + ndirections * (x + NX * y);
    }

public:
    // Constructor
    LBmethod(const unsigned int NSTEPS, const unsigned int NX, const unsigned int NY, const double u_lid, const double Re, const double rho0, const unsigned int num_cores);

    // Methods
    void Initialize();
    void Equilibrium();
    void UpdateMacro();
    void Collisions();
    void Streaming();
    void Run_simulation();
    void Visualization(unsigned int t);
};

#endif // LBMETHOD_H
