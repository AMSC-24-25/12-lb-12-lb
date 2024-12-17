#ifndef LBM_H
#define LBM_H

#include <vector>
#include <array>
#include <utility>
#include <string>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

#define INDEX(x, y, NX) ((x) + (NX) * (y)) // Convert 2D indices (x, y) into 1D index
#define INDEX3D(x, y, i, NX, ndirections) ((i) + (ndirections) * ((x) + (NX) * (y)))

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
    const std::array<std::pair<int, int>, 9> direction;
    const std::array<double, 9> weight;

    // Simulation data
    std::vector<double> rho;
    std::vector<std::pair<double, double>> u;
    std::vector<double> f_eq;
    std::vector<double> f;

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