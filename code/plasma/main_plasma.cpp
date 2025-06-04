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
    //
    // (a) Simulation domain (SI):
    const double Lx_SI = 1e-6;   // m
    const double Ly_SI = 1e-6;   // m

    // (b) Grid resolution:
    const size_t NX = 500;       // # nodes in x
    const size_t NY = 500;       // # nodes in y

    // (c) Number of time‐steps:
    const size_t NSTEPS = 1001;

    // (d) Ion parameters:
    const size_t Z_ion = 1;                   // e.g. H⁺
    const size_t A_ion = 1;                   // mass # = 1
    const double r_ion = 2.0e-10;             // ion radius [m]

    // (e) Time step in SI:
    const double dt_SI = 1e-12;               // 1/s

    // (f) Temperatures:
    const double T_e_SI = 700.0;              // electron temp [K]
    const double T_i_SI = 300.0;              // ion temp [K]

    // (g) External E‐field in SI [V/m]:
    const double Ex_SI = 1e5;     // 10⁵ V/m in x
    const double Ey_SI = 0.0;     // 0 V/m in y

    // (h) Choose Poisson solver and BC type:
    const PoissonType poisson_solver = PoissonType::SOR;
    // Options:
    // • NONE
    // • GAUSS_SEIDEL
    // • SOR
    // • FFT
    const BCType      bc_mode        = BCType::BOUNCE_BACK;
    // Options:
    // • PERIODIC
    // • BOUNCE_BACK
    const double      omega_sor      = 1.8;    // only used if SOR is selected

    // Define clock to evaluate time intervals
    const auto start_time = std::chrono::high_resolution_clock::now();

    //────────────────────────────────────────────────────────────────────────────
    // (3) Construct LBmethod:
    //────────────────────────────────────────────────────────────────────────────
    LBmethod lb(NSTEPS,
                NX, NY,
                n_cores,
                Z_ion, A_ion, r_ion,
                Lx_SI, Ly_SI,
                dt_SI,
                T_e_SI, T_i_SI,
                Ex_SI, Ey_SI,
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
    std::ofstream file("simulation_time_plasma_details.csv", std::ios::app); //Append mode
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open the output file." << std::endl;
        return 1;
    }

    // Write header if file is empty
    if (file.tellp() == 0) {
        file << "Grid_Dimension,Number_of_Steps,Number_of_Cores,Total_Computation_Time(ms)\n";
    }

    // Write details
    file << NX << "x" << NX << "," << NSTEPS << "," << n_cores << "," << total_time << "\n";

    file.close();
    
    return 0;
}
