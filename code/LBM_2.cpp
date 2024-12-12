#include <iostream>
#include <fstream>
#include <cstring> 
#include <vector>
#include <array>
#include <cmath>
#include <omp.h>


// Helper macro for 1D indexing from 2D or 3D coordinates
#define INDEX(x, y, NX) ((x) + (NX) * (y)) // Convert 2D indices (x, y) into 1D index
#define INDEX3D(x, y, i, NX, ndirections) ((i) + (ndirections) * ((x) + (NX) * (y)))

class LBmethod{
    private:
    //Parameters:
    const unsigned int NSTEPS;       // Number of timesteps to simulate
    const unsigned int NX;           // Number of nodes in the x-direction
    const unsigned int NY = NX;           // Number of nodes in the y-direction (square domain)
    const double u_lid;            // Lid velocity at the top boundary
    const double Re;             // Reynolds number
    const double L;                // Length of the cavity
    const double rho0;           // Initial uniform density at the start
    //Fixed Parameters:
    const unsigned int ndirections = 9;   // Number of directions (D2Q9 model has 9 directions)
    const double nu = (u_lid * L) / Re;  // Kinematic viscosity calculated using Re
    const double tau = 3.0 * nu + 0.5;    // Relaxation time for BGK collision model

    

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

    std::vector<double> rho; // Density 
    std::vector<std::pair<double, double>> u; // Velocity 
    std::vector<double> f_eq; // Equilibrium distribution function array
    std::vector<double> f; //  Distribution function array

    // Maximum number of threads
    const unsigned int n_threads= omp_get_max_threads();
    //omp_set_num_threads(n_threads);

    public:
    //Constructor:
    LBmethod(const unsigned int NSTEPS, const unsigned int NX, const double u_lid, const double Re, const double L, const double rho0 ): NSTEPS(NSTEPS), NX(NX), u_lid(u_lid), Re(Re), L(L), rho0(rho0){}
   
    //Methods:
    void Initialize(){
        //Vectors to store simulation data:
        rho.assign(NX * NY, rho0); // Density initialized to rho0 everywhere
        u.assign(NX * NY, {0.0, 0.0}); // Velocity initialized to 0
        f_eq.assign(NX * NY * ndirections, 0.0); // Equilibrium distribution function array
        f.assign(NX * NY * ndirections, 0.0); //  Distribution function array


        // Apply boundary condition: set velocity at the top lid (moving lid)
        for (unsigned int x = 0; x < NX; ++x) {
            unsigned int y = NY - 1; // Top boundary index
            u[INDEX(x, y, NX)].first = u_lid; // Set horizontal velocity to u_lid
            u[INDEX(x, y, NX)].second = 0.0;  // Vertical velocity is 0 at the top lid
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
                    std::memcpy(f.data(), f_eq.data(), f.size() * sizeof(double));
                }
            }
        }
        std::cout << "Equilibrium (initial state):\n";
        PrintDensity();
        PrintVelocity();
        PrintDistributionF();
        //unsigned int t=0;
        //Save_Output(t);
        
        
    }

    void UpdateMacro(){
        for (unsigned int x=0; x<NX; ++x){
            for (unsigned int y = 0; y < NY; ++y) {
                size_t idx = INDEX(x, y, NX);
                double rho_local = 0.0;
                double ux_local = 0.0;
                double uy_local = 0.0;
                for (unsigned int i = 0; i < ndirections; ++i) {
                    const double fi=f[INDEX3D(x, y, i, NX, ndirections)];
                    rho_local += fi;
                    ux_local += fi * direction[i].first;
                    uy_local += fi * direction[i].second;
                }
                if (rho_local > 1e-10) {
                    ux_local /= rho_local;
                    uy_local /= rho_local;
                }
                rho[idx]=rho_local;
                u[idx].first=ux_local;
                u[idx].second=uy_local;
                for (unsigned int i = 0; i < ndirections; ++i){
                    double ux = u[idx].first; // Horizontal velocity at point (x, y)
                    double uy = u[idx].second; // Vertical velocity at point (x, y)
                    double u2 = ux * ux + uy * uy; // Square of the speed magnitude
                    double cx = direction[i].first; // x-component of direction vector
                    double cy = direction[i].second; // y-component of direction vector
                    double cu = (cx * ux + cy * uy); // Dot product (c_i · u)
                    f_eq[INDEX3D(x, y, i, NX, ndirections)] = weight[i] * rho[idx] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
                } 
            }
        }
    }

    void Collisions(){
        //we use f=f-(f-f_eq)/tau from BGK
        for (unsigned int x=0;x<NX;++x){
            for (unsigned int y=0;y<NY;++y){
                for (unsigned int i=0;i<ndirections;++i){
                    f[INDEX3D(x, y, i, NX, ndirections)]=f[INDEX3D(x, y, i, NX, ndirections)]-(f[INDEX3D(x, y, i, NX, ndirections)]-f_eq[INDEX3D(x, y, i, NX, ndirections)])/tau;
                }
            }
        }
    }

    void Streaming(){
        //f(x,y,t+1)=f(x-cx,y-cy,t)
        std::vector<double> f_temp(NX * NY * ndirections, 0.0); // distribution function array temporaneal
        for (unsigned int x=0;x<NX;++x){
            for (unsigned int y=0;y<NY;++y){
                for (unsigned int i=0;i<ndirections;++i){
                    int x_str = x - direction[i].first;
                    int y_str = y - direction[i].second;
                    //check for particles inside the walls (sorta BC but they will be applyied more specifically after)
                    if(x_str<0) x_str=1;//bounceback (only position not velocity)
                    if(x_str>=NX) x_str=NX-1;//bounceback
                    if(y_str<0) y_str=1;//bounceback
                    if(y_str>=NY) y_str=NY-1;//bounceback
                    //apply straming function
                    f_temp[INDEX3D(x, y, i, NX, ndirections)] = f[INDEX3D(x_str, y_str, i, NX, ndirections)];
                }
            }
        }
        std::swap(f, f_temp);//f_temp is f at t=t+1 so now we use the new function f_temp in f
    }

    void BC() {
        //(since NX=NY we use only one outer for)
        for(unsigned int a=0; a<NX; ++a) {
            for(unsigned int i = 0; i < ndirections; ++i){
                //TOP BOUNDARY: (LID-VELOCITY)
                if (direction[i].second < 0) { 
                    //fi=f_opposite - 6*wi*rho*ci*u_lid --> from lattice Boltzamann equilibrium function and momentum conservation
                    size_t opp = INDEX3D(a, NY-1, (i + 4) % ndirections, NX, ndirections); // find the opposite direction of i
                    f[INDEX3D(a, NY-1, i, NX, ndirections)] = f[opp] - 6.0 * weight[i] * rho[INDEX(a,NY-1,NX)] * direction[i].first * u_lid;    
                }
                //BOTTOM BOUNDARY: (NO-SLIP)(Stationary Wall)
                if (direction[i].second> 0) { 
                    size_t opp = INDEX3D(a, 0, (i + 4) % ndirections, NX, ndirections); // find the opposite direction of i
                    f[INDEX3D(a, 0, i, NX, ndirections)] = f[opp];
                }
                //LEFT BOUNDARY:
                if (direction[i].first> 0) { 
                    size_t opp = INDEX3D(0, a, (i + 4) % ndirections, NX, ndirections);
                    f[INDEX3D(0, a, i, NX, ndirections)] = f[opp];
                }
                //RIGHT BOUNDARY:
                if (direction[i].first < 0) {
                    size_t opp = INDEX3D(NX-1, a, (i + 4) % ndirections, NX, ndirections);
                    f[INDEX3D(NX - 1, a, i, NX, ndirections)] = f[opp];
                }
            }
        }

    }
    
    void Run_simulation(){
        for (unsigned int t=0; t<NSTEPS; ++t){
            Collisions();
            Streaming();
            BC();
            UpdateMacro();
            if (t%2==0){
                Save_Output(t);
            }
            std::cout << "\n";
            std::cout << "Step: "+std::to_string(t+1)<< std::endl;
            PrintDensity();
            PrintVelocity();
            PrintDistributionF();
        }
    }


    void PrintDensity(){
        std::cout << "Density:\n";
        for (unsigned int x = 0; x < NX; ++x) {
            for (unsigned int y = 0; y < NY; ++y) {
                std::cout << std::fixed << rho[INDEX(x, y, NX)] << " ";
            }
            std::cout << "\n";
        }
    }
    
    void PrintVelocity(){
        std::cout << "Velocity:\n";
        for (unsigned int x = 0; x < NX; ++x) {
            for (unsigned int y = 0; y < NY; ++y) {
                std::cout << "(" << u[INDEX(x, y, NX)].first << ", " << u[INDEX(x, y, NX)].second << ") ";
            }
            std::cout << "\n";
        }
    }

    void PrintDistributionF(){
        // Print the computed f values for debugging purposes
        std::cout << "Distribution function:\n";
        for (unsigned int x = 0; x < NX; ++x) {
            for (unsigned int y = 0; y < NY; ++y) {
                size_t idx = INDEX(x, y, NX);
                std::cout << "Point (" << x << ", " << y << "): ";
                for (unsigned int i = 0; i < ndirections; ++i) {
                    std::cout << f[INDEX3D(x, y, i, NX, ndirections)] << " ";
                }
                std::cout << "\n";
            }
        }
    }

    void Save_Output(unsigned int t) {
        std::ofstream file("output_" + std::to_string(t) + ".csv");
        if (!file.is_open()) {
            std::cerr << "Errore: impossibile aprire il file." << std::endl;
            return;
        }

        file << "x,y,u_x,u_y,rho";
        for (unsigned int i = 0; i < ndirections; ++i) {
            file << ",f" << i;
        }
        file << "\n";

        for (unsigned int x = 0; x < NX; ++x) {
            for (unsigned int y = 0; y < NY; ++y) {
                size_t idx = INDEX(x, y, NX);
                file << x << "," << y << "," << u[idx].first << "," << u[idx].second<< "," << rho[idx] ;
                for (unsigned int i = 0; i < ndirections; ++i) {
                    file << "," << f[INDEX3D(x, y, i, NX, ndirections)];
                }
                file << "\n";
            }
        }
        file.close();
        std::cout << "File at t = " + std::to_string(t) + " saved" << std::endl;
    }
    
};

int main(){
    const unsigned int NSTEPS = 15;       // Number of timesteps to simulate
    const unsigned int NX = 5;           // Number of nodes in the x-direction
    const double u_lid = 1.0;            // Lid velocity at the top boundary
    const double Re = 100.0;             // Reynolds number
    const double L = 1.0;                // Length of the cavity
    const double rho = 1.0;             // Initial uniform density at the start

    LBmethod lb(NSTEPS,NX,u_lid,Re,L,rho);
    lb.Initialize();
    lb.Run_simulation();

    std::cout << "Simulation completed." << std::endl;
    return 0;
}
