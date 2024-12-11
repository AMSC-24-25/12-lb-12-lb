#include <iostream>
#include <vector>
#include <array>

// Helper macro for 1D indexing from 2D or 3D coordinates
#define INDEX(x, y, NX) ((x) + (NX) * (y)) // Convert 2D indices (x, y) into 1D index
#define INDEX3D(x, y, i, NX, ndirections) ((i) + (ndirections) * ((x) + (NX) * (y)))

int main() {
    const unsigned int NSTEPS = 10;       // Number of timesteps to simulate
    const unsigned int NX = 5;           // Number of nodes in the x-direction
    const unsigned int NY = NX;           // Number of nodes in the y-direction (square domain)
    const unsigned int ndirections = 9;   // Number of directions (D2Q9 model has 9 directions)
    const double u_lid = 1.0;            // Lid velocity at the top boundary
    const double Re = 100.0;             // Reynolds number
    const double L = 1.0;                // Length of the cavity
    const double nu = (u_lid * L) / Re;  // Kinematic viscosity calculated using Re
    const double tau = 3.0 * nu + 0.5;    // Relaxation time for BGK collision model
    const double rho0 = 1.0;             // Initial uniform density at the start

    // Define D2Q9 lattice directions (velocity directions for D2Q9 model)
    const std::array<std::pair<int, int>, 9> direction = {
        std::make_pair(0, 0),   // Rest direction
        std::make_pair(1, 0),   // Right
        std::make_pair(0, 1),   // Up
        std::make_pair(-1, 0),  // Left
        std::make_pair(0, -1),  // Down
        std::make_pair(1, 1),   // Top-right diagonal
        std::make_pair(-1, 1),  // Top-left diagonal
        std::make_pair(-1, -1), // Bottom-left diagonal
        std::make_pair(1, -1)   // Bottom-right diagonal
    };

    // D2Q9 lattice weights
    const std::array<double, 9> weight = {
        4.0 / 9.0,  // Weight for the rest direction
        1.0 / 9.0,  // Right
        1.0 / 9.0,  // Up
        1.0 / 9.0,  // Left
        1.0 / 9.0,  // Down
        1.0 / 36.0, // Top-right diagonal
        1.0 / 36.0, // Top-left diagonal
        1.0 / 36.0, // Bottom-left diagonal
        1.0 / 36.0  // Bottom-right diagonal
    };

    // Vectors to store simulation data
    std::vector<double> rho(NX * NY, rho0); // Density initialized to rho0 everywhere
    std::vector<std::pair<double, double>> u(NX * NY, {0.0, 0.0}); // Velocity initialized to 0
    std::vector<double> f_eq(NX * NY * ndirections, 0.0); // Equilibrium distribution function array

    //Print the density for debugging purposes
    std::cout << "Density:\n";
    for (unsigned int x = 0; x < NX; ++x) {
        for (unsigned int y = 0; y < NY; ++y) {
            std::cout << std::fixed << rho[INDEX(x, y, NX)] << " ";
        }
        std::cout << "\n";
    }

    // Apply boundary condition: set velocity at the top lid (moving lid)
    for (unsigned int x = 0; x < NX; ++x) {
        unsigned int y = NY - 1; // Top boundary index
        u[INDEX(x, y, NX)].first = u_lid; // Set horizontal velocity to u_lid
        u[INDEX(x, y, NX)].second = 0.0;  // Vertical velocity is 0 at the top lid
    }
    //Print the velocity for debugging purposes
    std::cout << "Velocity:\n";
    for (unsigned int x = 0; x < NX; ++x) {
        for (unsigned int y = 0; y < NY; ++y) {
            std::cout << "(" << u[INDEX(x, y, NX)].first << ", " << u[INDEX(x, y, NX)].second << ") ";
        }
        std::cout << "\n";
    }

    // Compute the equilibrium distribution function f_eq
    for (unsigned int x = 0; x < NX; ++x) {
        for (unsigned int y = 0; y < NY; ++y) {
            size_t idx = INDEX(x, y, NX); // Get 1D index for 2D point (x, y)
            double ux = u[idx].first; // Horizontal velocity at point (x, y)
            double uy = u[idx].second; // Vertical velocity at point (x, y)
            double u2 = ux * ux + uy * uy; // Square of the speed magnitude

            for (unsigned int i = 0; i < ndirections; ++i) {
                double cx = direction[i].first; // x-component of direction vector
                double cy = direction[i].second; // y-component of direction vector
                double cu = (cx * ux + cy * uy); // Dot product (c_i · u)

                // Compute f_eq using the BGK collision formula
                f_eq[INDEX3D(x, y, i, NX, ndirections)] = weight[i] * rho[idx] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
            }
        }
    }

    // Print the computed f_eq values for debugging purposes
    std::cout << "Equilibrium distribution f_eq:\n";
    for (unsigned int x = 0; x < NX; ++x) {
        for (unsigned int y = 0; y < NY; ++y) {
            size_t idx = INDEX(x, y, NX);
            std::cout << "Point (" << x << ", " << y << "): ";
            for (unsigned int i = 0; i < ndirections; ++i) {
                std::cout << f_eq[INDEX3D(x, y, i, NX, ndirections)] << " ";
            }
            std::cout << "\n";
        }
    }
    //The following function are unusful in the case of equilibrium, however are useful after when they give different results
    //Calculate the density and velocity
    std::vector<std::pair<double, double>> u_new(NX * NY, {0.0, 0.0});
    std::vector<double> rho_new(NX * NY, 0);
    for (unsigned int x = 0; x < NX; ++x) {
        for (unsigned int y = 0; y < NY; ++y) {
            size_t idx = INDEX(x, y, NX);
            for (unsigned int i = 0; i < ndirections; ++i) {
                double fi = f_eq[INDEX3D(x, y, i, NX, ndirections)];
                rho_new[idx] += fi;
                u_new[idx].first += fi * direction[i].first;
                u_new[idx].second += fi * direction[i].second;
            }

            if (rho_new[idx] > 1e-10) {
                u_new[idx].first /= rho_new[idx];
                u_new[idx].second /= rho_new[idx];
            }
        }
    }
    //Print the density for debugging purposes
    std::cout << "Density:\n";
    for (unsigned int x = 0; x < NX; ++x) {
        for (unsigned int y = 0; y < NY; ++y) {
            std::cout << std::fixed << rho_new[INDEX(x, y, NX)] << " ";
        }
        std::cout << "\n";
    }

    //Print the velocity for debugging purposes
    std::cout << "Velocity:\n";
    for (unsigned int x = 0; x < NX; ++x) {
        for (unsigned int y = 0; y < NY; ++y) {
            std::cout << "(" << u_new[INDEX(x, y, NX)].first << ", " << u_new[INDEX(x, y, NX)].second << ") ";
        }
        std::cout << "\n";
    }
    
    

    return 0; // End of simulation
}
