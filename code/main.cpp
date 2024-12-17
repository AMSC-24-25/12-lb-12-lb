#include "LBM.hpp"
#include <iostream>
#include <chrono>
#include <fstream>

int main(int argc, char* argv[]) {
    const unsigned int NSTEPS = 1000;       // Number of timesteps to simulate
    const unsigned int NX = 128;           // Number of nodes in the x-direction
    const unsigned int NY = NX;           // Number of nodes in the y-direction
    const double u_lid = 0.1;            // Lid velocity at the top boundary
    const double Re = 100.0;             // Reynolds number
    const double rho = 1.0;             // Initial uniform density at the start
    unsigned int ncores = std::stoi(argv[1]); // Take the number of cores from the first argument


    auto start_time = std::chrono::high_resolution_clock::now();

    LBmethod lb(NSTEPS, NX, NY, u_lid, Re, rho, ncores);
    lb.Initialize();
    lb.Run_simulation();

    //Measure end time
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    // Write computational details to CSV
    std::ofstream file("simulation_time_details.csv", std::ios::app); //Append mode
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open the output file." << std::endl;
        return 1;
    }

    // Write header if file is empty
    if (file.tellp() == 0) {
        file << "Grid_Dimension,Number_of_Steps,Number_of_Cores,Total_Computation_Time(s)\n";
    }

    // Write details
    file << NX << "x" << NX << "," << NSTEPS << "," << ncores << "," << total_time << "\n";

    file.close();
    

    std::cout << "Simulation completed. Use ffmpeg to generate a video:" << std::endl;
    std::cout << "ffmpeg -framerate 10 -i frames/frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p simulation.mp4" << std::endl;
    return 0;
}
