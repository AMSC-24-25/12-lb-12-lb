#include "plasma.hpp"

#include <iostream>
#include <chrono>
#include <fstream>

int main(int argc, char* argv[]) {
    // (1) Number of OpenMP threads 
    const size_t n_cores = std::stoi(argv[1]); // Take the number of cores from the first argument

    //────────────────────────────────────────────────────────────────────────────
    // (2) User‐Defined Physical (SI) Parameters
    //────────────────────────────────────────────────────────────────────────────

    // (b) Grid resolution:
    const int NX = 200;       // # nodes in x
    const int NY = 200;       // # nodes in y

    // (c) Number of time‐steps:
    const int NSTEPS = 200; // total number of time steps

    // (d) Ion parameters:
    const int Z_ion = 1;                   // atomic number
    const int A_ion = 1;                   // mass number
    
    // (e) Physical denisty:
    const double n_e_SI_init = 1e11;  // Initial physical density [m⁻³]
    const double n_n_SI_init = 1e18;

    // (f) Physical Temperatures:
    const double T_e_SI_init = 1e4;              // Initial electron temp [K]
    const double T_i_SI_init = 300;              // Initial ion temp [K]
    const double T_n_SI_init = 300;

    // (g) External E‐field in SI [V/m]:
    const double Ex_SI = 1e-2;     // External electric field along x [V/m]
    const double Ey_SI = 0.0;     // External electric field along y [V/m]

    // (h) Choose Poisson solver and BC type:
    const poisson::PoissonType poisson_solver = poisson::PoissonType::FFT;
    // Options:
    // • NONE
    // • GS
    // • SOR
    // • FFT
    // • NPS
    const streaming::BCType      bc_mode        = streaming::BCType::Periodic;
    // Options:
    // • Periodic
    // • BounceBack
    const double      omega_sor      = 1.8;    // only used if SOR is selected

    // Define clock to evaluate time intervals
    const auto start_time = std::chrono::high_resolution_clock::now();

    //────────────────────────────────────────────────────────────────────────────
    // (3) Construct LBmethod:
    //────────────────────────────────────────────────────────────────────────────
    LBmethod lb(NSTEPS,
                NX, NY,
                n_cores,
                Z_ion, A_ion,
                Ex_SI, Ey_SI,
                T_e_SI_init, T_i_SI_init, T_n_SI_init,
                n_e_SI_init, n_n_SI_init,
                poisson_solver,
                bc_mode,
                omega_sor);

    //────────────────────────────────────────────────────────────────────────────
    // (4) Run the simulation:
    //────────────────────────────────────────────────────────────────────────────
    lb.Run_simulation();
  
    //Measure end time
    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Write computational details to CSV
    std::ofstream file("build/simulation_time_plasma_details.csv", std::ios::app); //Append mode
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open the output file." << std::endl;
        return 1;
    }

    // Write header if file is empty
    if (file.tellp() == 0) {
        file << "Grid_Dimension,Number_of_Steps,Number_of_Cores,Poisson,BC,Total_Computation_Time(ms)\n";
    }

    // Write details
    file << NX << "x" << NY << "," << NSTEPS << "," << n_cores << "," << static_cast<int>(poisson_solver) << "," << static_cast<int>(bc_mode) << "," << total_time << "\n";

    file.close();
    
    return 0;
}
