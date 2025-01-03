#include "plasma.hpp"
#include <iostream>
#include <chrono>
#include <fstream>

int main(int argc, char* argv[]) {
    const size_t NSTEPS = 1000;       // Number of timesteps to simulate
    const size_t NX = 129;           // Number of nodes in the x-direction
    const size_t NY = NX;           // Number of nodes in the y-direction
    const double Re = 100.0;             // Reynolds number
    const double tau_ion=0.01;
    const double tau_el=0.01;
    const size_t ncores = std::stoi(argv[1]); // Take the number of cores from the first argument

    const auto start_time = std::chrono::high_resolution_clock::now();

    LBmethod lb(NSTEPS, NX, NY, Re, ncores, tau_ion, tau_el);
    lb.Initialize();
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
    file << NX << "x" << NX << "," << NSTEPS << "," << ncores << "," << total_time << "\n";

    file.close();
    

    return 0;
}

