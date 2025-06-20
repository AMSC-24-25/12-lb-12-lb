#include <iostream>
#include <fstream>
#include <cstring> 
#include <string>
#include <filesystem>
#include <vector>
#include <array>
#include <cmath>
#include <omp.h>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace fs=std::filesystem;
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
    const double rho0;           // Initial uniform density at the start
    double u_lid_dyn;
    const unsigned int num_cores; // Number of threads for parallelization
    //Fixed Parameters:
    const unsigned int ndirections = 9;   // Number of directions (D2Q9 model has 9 directions)
    const double nu = (u_lid * NY) / Re;  // Kinematic viscosity calculated using Re
    const double tau = 3.0 * nu + 0.5;    // Relaxation time for BGK collision model
    const double sigma = 10.0; //this is a parameter that determines the slope of how fast u_lid_dyn tends to u_lid

    

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

    public:
    //Constructor:
    public:
    LBmethod(const unsigned int NSTEPS, const unsigned int NX, const double u_lid, const double Re, const double rho0, const unsigned int num_cores)
        : NSTEPS(NSTEPS), NX(NX), u_lid(u_lid), Re(Re), rho0(rho0), num_cores(num_cores) {}
    //Methods:
    void Initialize(){
        //Vectors to store simulation data:
        rho.assign(NX * NY, rho0); // Density initialized to rho0 everywhere
        u.assign(NX * NY, {0.0, 0.0}); // Velocity initialized to 0
        f_eq.assign(NX * NY * ndirections, 0.0); // Equilibrium distribution function array
        f.assign(NX * NY * ndirections, 0.0); //  Distribution function array


        Equilibrium();//first equilibrium condition with rho=1 and u=0
        //iniitialize the distribution function as a static one
        #pragma omp parallel for collapse(2)
        for(unsigned int x=0;x<NX;++x){
            for (unsigned int y=0;y<NY;++y){
                for (unsigned int i=0;i<ndirections;++i){
                    f[INDEX3D(x,y,i,NX,ndirections)]=weight[i];
                }
            }
        }

        if (NX<=6){
            std::cout << "Equilibrium (initial state):\n";
            PrintDensity();
            PrintVelocity();
            PrintDistributionF(); 
            PrintDistributionEquilibrium();
        }
         
    }

    void Equilibrium(){
        // Compute the equilibrium distribution function f_eq
        //collapse(2) combines the 2 loops into a single iteration space-> convient when I have large Nx and Ny (not when they're really different tho)
        //static ensure uniform distribution
        //I don't do collapse(3) because the inner loop is light
        #pragma omp parallel for collapse(2) schedule(static)
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

    }

    void UpdateMacro(){
        #pragma omp parallel for collapse(2) schedule(static)
        //or schedule(dynamic, chunk_size) if the computational complexity varies
        for (unsigned int x=0; x<NX; ++x){
            for (unsigned int y = 0; y < NY; ++y) {
                size_t idx = INDEX(x, y, NX);
                double rho_local = 0.0;
                double ux_local = 0.0;
                double uy_local = 0.0;

                #pragma omp parallel for reduction(+:rho_local, ux_local, uy_local)
                for (unsigned int i = 0; i < ndirections; ++i) {
                    const double fi=f[INDEX3D(x, y, i, NX, ndirections)];
                    rho_local += fi;
                    ux_local += fi * direction[i].first;
                    uy_local += fi * direction[i].second;
                }
                if (rho_local<1e-10){
                    rho[idx] = 0.0;
                    ux_local = 0.0;
                    uy_local = 0.0;
                }
                else {
                    rho[idx] = rho_local;
                    ux_local /= rho_local;
                    uy_local /= rho_local;
                }
                u[INDEX(x, y, NX)].first=ux_local;
                u[INDEX(x, y, NX)].second=uy_local;
            }
        }
        Equilibrium();
    }

    void Collisions(){
        #pragma omp parallel for collapse(2) schedule(static)
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
        std::vector<int> opposites = {0, 3, 4, 1, 2, 7, 8, 5, 6}; //Opposite velocities
        std::vector<double> f_temp(NX * NY * ndirections, 0.0); // distribution function array temporaneal

        //paralleliation only in the bulk streaming
        //Avoid at boundaries to prevent race conditions
        #pragma omp parallel for collapse(2) schedule(static)
        for (unsigned int x=0;x<NX;++x){
            for (unsigned int y=0;y<NY;++y){
                for (unsigned int i=0;i<ndirections;++i){

                    int x_str = x - direction[i].first;
                    int y_str = y - direction[i].second;
                    //streaming process
                    if(x_str >= 0 && x_str < NX && y_str >= 0 && y_str < NY){
                        f_temp[INDEX3D(x,y,i,NX,ndirections)]=f[INDEX3D(x_str,y_str,i,NX,ndirections)];
                    }
                }
            }
        }
        //BCs
        //Sides + bottom angles
        //Left and right //maybe can be merged
        for (unsigned int y=0;y<NY;++y){
            //Left
            for (unsigned int i : {3,6,7}){//directions: left, top left, bottom left
                f_temp[INDEX3D(0,y,opposites[i],NX,ndirections)]=f[INDEX3D(0,y,i,NX,ndirections)];
            }
            //Right
            for (unsigned int i : {1,5,8}){//directions: right, top right, top left
                f_temp[INDEX3D(NX-1,y,opposites[i],NX,ndirections)]=f[INDEX3D(NX-1,y,i,NX,ndirections)];
            }
        }
        //Bottom
        for (unsigned int x=0;x<NX;++x){
            for (unsigned int i : {4,7,8}){//directions: bottom, bottom left, bottom right
                f_temp[INDEX3D(x,0,opposites[i],NX,ndirections)]=f[INDEX3D(x,0,i,NX,ndirections)];
            }
        }
        //Top
        for (unsigned int x=0;x<NX;++x){
            //since we are using density we can either recompute all the macroscopi quatities before or compute rho_local
            double rho_local=0.0;
            for (unsigned int i=0;i<ndirections;++i){
                rho_local+=f[INDEX3D(x,NY-1,i,NX,ndirections)];
            }
            for (unsigned int i : {2,5,6}){//directions: up,top right, top left
                //this is the expresion of -2*w*rho*dot(c*u_lid)/cs^2 since cs^2=1/3 and also u_lid=(0.1,0)
                double deltaf=-6.0*weight[i]*rho_local*(direction[i].first*u_lid_dyn);
                f_temp[INDEX3D(x, NY-1, opposites[i], NX, ndirections)] = f[INDEX3D(x,NY-1,i,NX,ndirections)] + deltaf;
            }
        }

        std::swap(f, f_temp);//f_temp is f at t=t+1 so now we use the new function f_temp in f
    }

    
    void Run_simulation(){

        // Set threads for this simulation
        omp_set_num_threads(num_cores);

        // Ensure the directory for frames exists
        std::string frame_dir = "frames";
        if (!fs::exists(frame_dir)) {
            fs::create_directory(frame_dir);
            std::cout << "Directory created for frames: " << frame_dir << std::endl;
        }

        for (unsigned int t=0; t<NSTEPS; ++t){
            if (double(t)<sigma){
                u_lid_dyn = u_lid*double(t)/sigma;
            }
            else{
                u_lid_dyn = u_lid;
            }
            
            Collisions();
            Streaming();
            UpdateMacro();

            if (t%1==0){
                Visualization(t);
            }
            if (NSTEPS<10 && NX<=6){
                Save_Output(t);
                std::cout << "\n";
                std::cout << "Step: "+std::to_string(t+1)<< std::endl;
                std::cout<<"lid velocity= "<<u_lid_dyn<<std::endl;
                PrintDensity();
                PrintVelocity();
                PrintDistributionF();
                PrintDistributionEquilibrium();
            }
        }
        ConfrontData();
    }


    void PrintDensity(){
        std::cout << "Density:\n";
        const int width = 12; // Adjust for number size
        std::cout << std::fixed << std::setprecision(6); // Fixed decimal precision
        for (unsigned int y = 1; y <= NY; ++y) {
            for (unsigned int x = 0; x < NX; ++x) {
                std::cout << std::setw(width) << rho[INDEX(x, NY-y, NX)] << ", ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    
    void PrintVelocity(){
        std::cout << "Velocity:\n";
        const int width = 12; // Adjust for number size
        std::cout << std::fixed << std::setprecision(6); // Fixed decimal precision
        for (unsigned int y = 1; y <= NY; ++y) {
            for (unsigned int x = 0; x < NX; ++x) {
                std::cout << "(" << std::setw(width) << u[INDEX(x, NY-y, NX)].first << ", " << std::setw(width) << u[INDEX(x, NY-y, NX)].second << ") ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    void PrintDistributionF(){
        // Print the computed f values for debugging purposes
        std::cout << "Distribution function:\n";
        const int width = 12; // Adjust for number size
        std::cout << std::fixed << std::setprecision(6); // Fixed decimal precision
        // Define block layout: indices for each block
        int block_layout[3][3] = {
            {6, 2, 5}, // Top row: M6, M2, M5
            {3, 0, 1}, // Middle row: M3, M0, M1
            {7, 4, 8}  // Bottom row: M7, M4, M8
        };
        // Iterate over rows in the block layout
        for (int row = 0; row < 3; ++row) { 
            // Print each row of the block layout
            for (int y = NY - 1; y >= 0; --y) { // Iterate through the y-coordinates
                for (int col = 0; col < 3; ++col) { 
                    int i = block_layout[row][col]; // Get the direction index for the block
                    for (unsigned int x = 0; x < NX; ++x) { // Iterate over x-coordinates
                        std::cout << std::setw(width) << f[INDEX3D(x, y, i, NX, ndirections)] << " ";
                    }
                    std::cout << " | "; // Separator between blocks in a row
                }
                std::cout << "\n"; // End of row for the block
            }
            std::cout << std::string(3 * (width * NX + 3 * NX) + 4, '-') << "\n"; // Horizontal divider
        }
        std::cout << "\n";
    }

    void PrintDistributionEquilibrium(){
        // Print the computed f values for debugging purposes
        std::cout << "Equilibrium distribution function:\n";
        const int width = 12; // Adjust for number size
        std::cout << std::fixed << std::setprecision(6); // Fixed decimal precision
        // Define block layout: indices for each block
        int block_layout[3][3] = {
            {6, 2, 5}, // Top row: M6, M2, M5
            {3, 0, 1}, // Middle row: M3, M0, M1
            {7, 4, 8}  // Bottom row: M7, M4, M8
        };
        // Iterate over rows in the block layout
        for (int row = 0; row < 3; ++row) { 
            // Print each row of the block layout
            for (int y = NY - 1; y >= 0; --y) { // Iterate through the y-coordinates
                for (int col = 0; col < 3; ++col) { 
                    int i = block_layout[row][col]; // Get the direction index for the block
                    for (unsigned int x = 0; x < NX; ++x) { // Iterate over x-coordinates
                        std::cout << std::setw(width) << f_eq[INDEX3D(x, y, i, NX, ndirections)] << " ";
                    }
                    std::cout << " | "; // Separator between blocks in a row
                }
                std::cout << "\n"; // End of row for the block
            }
            std::cout << std::string(3 * (width * NX + 3 * NX) + 4, '-') << "\n"; // Horizontal divider
        }
        std::cout << "\n";
    }

    void Visualization(unsigned int t){
        static cv::Mat velocity_magn_mat, density_mat;
        static cv::Mat velocity_magn_norm, density_norm;
        static cv::Mat velocity_heatmap, density_heatmap;

        // Initialize only when t == 0
        if (t == 0) {
        // Initialize the heatmaps with the same size as the grid
            //OpenCV uses a row-major indexing
            velocity_magn_mat = cv::Mat(NY, NX, CV_32F);
            density_mat = cv::Mat(NY, NX, CV_32F);
        
            // Create matrices for normalized values
            velocity_magn_norm = cv::Mat(NY, NX, CV_32F);
            density_norm = cv::Mat(NY, NX, CV_32F);

            // Create heatmap images (8 bit images)
            velocity_heatmap = cv::Mat(NY, NX, CV_8UC3);
            density_heatmap = cv::Mat(NY, NX, CV_8UC3);
        }

        // Fill matrices with new data
        #pragma omp parallel for collapse(2)
        for (unsigned int x = 0; x < NX; ++x) {
            for (unsigned int y = 0; y < NY; ++y) {
                size_t idx = INDEX(x, y, NX);
                double ux = u[idx].first;
                double uy = u[idx].second;
                velocity_magn_mat.at<float>(y, x) = std::sqrt(ux * ux + uy * uy);
                density_mat.at<float>(y, x) = static_cast<float>(rho[idx]);
            }
        }

        // Normalize the matrices to 0-255 for display
        cv::normalize(velocity_magn_mat, velocity_magn_norm, 0, 255, cv::NORM_MINMAX);
        cv::normalize(density_mat, density_norm, 0, 255, cv::NORM_MINMAX);

        //8-bit images
        velocity_magn_norm.convertTo(velocity_magn_norm, CV_8U);
        density_norm.convertTo(density_norm, CV_8U);

        // Apply color maps
        cv::applyColorMap(velocity_magn_norm, velocity_heatmap, cv::COLORMAP_PLASMA);
        cv::applyColorMap(density_norm, density_heatmap, cv::COLORMAP_VIRIDIS);

        //Flip the image vertically (OpenCV works in the opposite way than our code)
        cv::flip(velocity_heatmap, velocity_heatmap, 0); //flips along the x axis
        cv::flip(density_heatmap, density_heatmap, 0);

        // Combine both heatmaps horizontally
        cv::Mat combined;
        cv::hconcat(velocity_heatmap, density_heatmap, combined);

        if(NSTEPS<=300){
            // Display the updated frame in a window
            cv::imshow("Velocity (Left) and Density (Right)", combined);
            cv::waitKey(1); // 1 ms delay for real-time visualization
        }
        
        // Save the current frame to a file
        std::string filename = "frames/frame_" + std::to_string(t) + ".png";
        cv::imwrite(filename, combined);
        
        
    }

    void Save_Output(unsigned int t) {

        //Define subfolder name
        std::string folder_name="output_files";

        //Create subfolder
        if (!fs::exists(folder_name)){
            fs::create_directory(folder_name);
        }

        //Construct file path
        std::string file_path= folder_name + "/file_" + std::to_string(t)+ ".cvs";
        std::ofstream file(file_path);
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

    void ConfrontData() {
        unsigned int x_c = int(NX / 2);  // Central x-coordinate
        unsigned int y_c = NY / 2;  // Central x-coordinate
        std::string output_file = "velocity_along_center.csv";
        std::ofstream file(output_file);

        if (!file.is_open()) {
            std::cerr << "Error: Unable to open the output file." << std::endl;
            return;
        }
        file<< "Vertical line:\n";
        file << "y,velocity_magnitude\n";

        for (unsigned int y = 0; y < NY; ++y) {
            size_t idx = INDEX(x_c, NY-y, NX);
            double velocity_magnitude = u[idx].second/u_lid;

            file << NY-y << "," << velocity_magnitude << "\n";
        }

        file<<"\n\n";
        file<< "Horizontal line:\n";
        file << "x,velocity_magnitude\n";

        for (unsigned int x = 0; x < NX; ++x) {
            size_t idx = INDEX(NX-x, y_c, NX);
            double velocity_magnitude = u[idx].first/u_lid;

            file << NX-x << "," << velocity_magnitude << "\n";
        }

        file.close();
        std::cout << "Velocity magnitudes along the central line have been written to: " << output_file << std::endl;
    }
    
};

int main(int argc, char* argv[]){
    const unsigned int NSTEPS = 1000;       // Number of timesteps to simulate
    const unsigned int NX = 128;           // Number of nodes in the x-direction
    const double u_lid = 0.1;            // Lid velocity at the top boundary
    const double Re = 100.0;             // Reynolds number
    const double rho = 1.0;             // Initial uniform density at the start
    unsigned int ncores = std::stoi(argv[1]); // Take the number of cores from the first argument

    // Measure the start time
    auto start_time = std::chrono::high_resolution_clock::now();

    LBmethod lb(NSTEPS,NX,u_lid,Re,rho,ncores);
    lb.Initialize();
    lb.Run_simulation();
    // Measure the end time
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    // Write computational details to CSV
    std::string output_file = "simulation_time_details.csv";
    std::ofstream file(output_file, std::ios::app); // Append mode
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
