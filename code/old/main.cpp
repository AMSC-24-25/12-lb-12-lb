#include "LBM.hpp"
#include <iostream>
#include <chrono>
#include <fstream>

int main(int argc, char* argv[]) {
    const unsigned int NSTEPS = 1000;       // Number of timesteps to simulate
    const unsigned int NX = 100;           // Number of nodes in the x-direction
    const unsigned int NY = NX;           // Number of nodes in the y-direction
    const double u_lid = 0.1;            // Lid velocity at the top boundary
    const double Re = 100.0;             // Reynolds number
    const double rho = 1.0;             // Initial uniform density at the start

    const auto start_time = std::chrono::high_resolution_clock::now();

    LBmethod lb(NSTEPS, NX, NY, u_lid, Re, rho);
    lb.Initialize();
    lb.Run_simulation();

    //Measure end time
    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Write computational details to CSV
    std::ofstream file("simulation_time_details.csv", std::ios::app); //Append mode
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open the output file." << std::endl;
        return 1;
    }

    // Write header if file is empty
    if (file.tellp() == 0) {
        file << "Grid_Dimension,Number_of_Steps,Number_of_Cores,Total_Computation_Time(ms)\n";
    }

    // Write details
    file << NX << "x" << NX << "," << NSTEPS << ",1," << total_time << "\n";

    file.close();
    

    return 0;
}
