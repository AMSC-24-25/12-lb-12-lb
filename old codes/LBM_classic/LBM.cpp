#include "LBM.hpp"
#include <cmath>
#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>


// Constructor
LBmethod::LBmethod(const size_t NSTEPS, const size_t NX,const size_t NY,  const double u_lid, const double Re, const size_t num_cores)
    : NSTEPS(NSTEPS), NX(NX), NY(NY), u_lid(u_lid), Re(Re), num_cores(num_cores), tau(3.0 * ((u_lid * NY) / Re) + 0.5), 
      directionx({0,1,0,-1,0,1,-1,-1,1}),
      directiony({0,0,1,0,-1,1,1,-1,-1}),  
      weight({  4.0 / 9.0, 
                1.0 / 9.0,
                1.0 / 9.0, 
                1.0 / 9.0, 
                1.0 / 9.0,
                1.0 / 36.0, 
                1.0 / 36.0, 
                1.0 / 36.0, 
                1.0 / 36.0}) {}


void LBmethod::Initialize() {
    //Vectors to store simulation data:
    rho.assign(NX * NY, 1.0); // Density initialized to rho0 everywhere
    ux.assign(NX * NY, 0.0); // Velocity initialized to 0
    uy.assign(NX * NY, 0.0); // Velocity initialized to 0
    f_eq.assign(NX * NY * ndirections, 0.0); // Equilibrium distribution function array
    f.assign(NX * NY * ndirections, 0.0); //  Distribution function array
    f_temp.assign(NX * NY * ndirections, 0.0);
    
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < NX * NY * ndirections; ++idx) {
        const size_t i = idx % ndirections; // Use size_t for consistency
        f[idx] = f_eq[idx] = weight[i];
    }

}

void LBmethod::Equilibrium() {
    // Compute the equilibrium distribution function f_eq
    #pragma omp parallel for collapse(2) schedule(static)
        for (size_t x = 0; x < NX; ++x) {
            for (size_t y = 0; y < NY; ++y) {
                const size_t idx = INDEX(x, y, NX); // Get 1D index for 2D point (x, y)
                const double u2 = ux[idx] * ux[idx] + uy[idx] * uy[idx]; // Square of the speed magnitude

                for (size_t i = 0; i < ndirections; ++i) {
                    const double cu = (directionx[i] * ux[idx] + directiony[i] * uy[idx]); // Dot product (c_i · u)

                    // Compute f_eq using the BGK collision formula
                    f_eq[INDEX(x, y, i, NX, ndirections)] = weight[i] * rho[idx] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
                }
            }
        }
}

void LBmethod::UpdateMacro() {
    #pragma omp parallel for collapse(2)
        for (size_t x=0; x<NX; ++x){
            for (size_t y = 0; y < NY; ++y) {
                const size_t idx = INDEX(x, y, NX);
                double rho_local = 0.0;
                double ux_local = 0.0;
                double uy_local = 0.0;


                for (size_t i = 0; i < ndirections; ++i) {
                    const double fi=f[INDEX(x, y, i, NX, ndirections)];
                    rho_local += fi;
                    ux_local += fi * directionx[i];
                    uy_local += fi * directiony[i];
                }
                if (rho_local<1e-10){
                    rho[idx] = 0.0;
                    ux[idx] = 0.0;
                    uy[idx] = 0.0;
                }
                else {
                    rho[idx] = rho_local;
                    ux[idx]=ux_local/rho_local;
                    uy[idx]=uy_local/rho_local;
                }
            }
        }
        Equilibrium();
}

void LBmethod::Collisions() {
        //we use f=f-(f-f_eq)/tau from BGK
        #pragma omp parallel for collapse(3)
        for (size_t x=0;x<NX;++x){
            for (size_t y=0;y<NY;++y){
                for (size_t i=0;i<ndirections;++i){
                    const size_t idx=INDEX(x, y, i, NX, ndirections);
                    f[idx]=f[idx]-(f[idx]-f_eq[idx])/tau;
                }
            }
        }
}

void LBmethod::Streaming() {
    //f(x,y,t+1)=f(x-cx,y-cy,t)
    // Parallelize the streaming step (over x and y)
    #pragma omp parallel
    {
        #pragma omp for collapse(3) schedule(static)
        for (size_t x = 0; x < NX; ++x) {
            for (size_t y = 0; y < NY; ++y) {
                for (size_t i = 0; i < ndirections; ++i) {
                    const int x_str = x - directionx[i];
                    const int y_str = y - directiony[i];
                    if (x_str >= 0 && x_str < NX && y_str >= 0 && y_str < NY) {
                        f_temp[INDEX(x, y, i, NX, ndirections)] = f[INDEX(x_str, y_str, i, NX, ndirections)];
                    }
                }
            }
        }
        // Parallelize the boundary conditions
        // Sides + bottom angles
        #pragma omp for schedule(static)
        for (size_t y = 0; y < NY; ++y) {
            // Left
            f_temp[INDEX(0, y, 1, NX, ndirections)] = f[INDEX(0, y, 3, NX, ndirections)];
            f_temp[INDEX(0, y, 8, NX, ndirections)] = f[INDEX(0, y, 6, NX, ndirections)];
            f_temp[INDEX(0, y, 5, NX, ndirections)] = f[INDEX(0, y, 7, NX, ndirections)];
            // Right
            f_temp[INDEX(NX - 1, y, 3, NX, ndirections)] = f[INDEX(NX - 1, y, 1, NX, ndirections)];
            f_temp[INDEX(NX - 1, y, 7, NX, ndirections)] = f[INDEX(NX - 1, y, 5, NX, ndirections)];
            f_temp[INDEX(NX - 1, y, 6, NX, ndirections)] = f[INDEX(NX - 1, y, 8, NX, ndirections)];
        }

        
        #pragma omp for schedule(static)
        for (size_t x = 0; x < NX; ++x) {
            // Bottom boundary conditions
            f_temp[INDEX(x, 0, 2, NX, ndirections)] = f[INDEX(x, 0, 4, NX, ndirections)];
            f_temp[INDEX(x, 0, 5, NX, ndirections)] = f[INDEX(x, 0, 7, NX, ndirections)];
            f_temp[INDEX(x, 0, 6, NX, ndirections)] = f[INDEX(x, 0, 8, NX, ndirections)];

            // Top boundary conditions (including computation of rho_local)
            double rho_local = 0.0;
            for (size_t i = 0; i < ndirections; ++i) {
                rho_local += f[INDEX(x, NY - 1, i, NX, ndirections)];
            }

            const double deltaf2 = -6.0 * weight[2] * rho_local * (directionx[2] * u_lid_dyn);
            const double deltaf5 = -6.0 * weight[5] * rho_local * (directionx[5] * u_lid_dyn);
            const double deltaf6 = -6.0 * weight[6] * rho_local * (directionx[6] * u_lid_dyn);

            f_temp[INDEX(x, NY - 1, 4, NX, ndirections)] = f[INDEX(x, NY - 1, 2, NX, ndirections)] + deltaf2;
            f_temp[INDEX(x, NY - 1, 7, NX, ndirections)] = f[INDEX(x, NY - 1, 5, NX, ndirections)] + deltaf5;
            f_temp[INDEX(x, NY - 1, 8, NX, ndirections)] = f[INDEX(x, NY - 1, 6, NX, ndirections)] + deltaf6;
        }
    }

    std::swap(f, f_temp);//f_temp is f at t=t+1 so now we use the new function f_temp in f
}

void LBmethod::Run_simulation() {
    
    // Set threads for this simulation
        omp_set_num_threads(num_cores);
        const double u_lid_over_sigma = u_lid / sigma;//precompute constant value
        // VideoWriter setup
        const std::string video_filename = "simulation.mp4";
        const double fps = 10.0; // Frames per second for the video
        video_writer.open(video_filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(NX, NY));

        if (!video_writer.isOpened()) {
            std::cerr << "Error: Could not open the video writer." << std::endl;
            return;
        }
    
        for (size_t t=0; t<NSTEPS; ++t){
            const double t_double = static_cast<double>(t);//avoid repeated type cast
            u_lid_dyn = (t_double < sigma) ? (u_lid_over_sigma * t_double) : u_lid;//replaced condition branching
            
            Collisions();
            Streaming();
            UpdateMacro();
            Visualization(t);
        }
    
        video_writer.release();
        std::cout << "Video saved as " << video_filename << std::endl;
}

void LBmethod::Visualization(size_t t) {
        static cv::Mat velocity_magn_mat;
        static cv::Mat velocity_magn_norm;
        static cv::Mat velocity_heatmap;

        // Initialize only when t == 0
        if (t == 0) {
        // Initialize the heatmaps with the same size as the grid
            //OpenCV uses a row-major indexing
            velocity_magn_mat = cv::Mat(NY, NX, CV_32F);
        
            // Create matrices for normalized values
            velocity_magn_norm = cv::Mat(NY, NX, CV_32F);

            // Create heatmap images (8 bit images)
            velocity_heatmap = cv::Mat(NY, NX, CV_8UC3);
        }

        // Fill matrices with new data
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t x = 0; x < NX; ++x) {
            for (size_t y = 0; y < NY; ++y) {
                const size_t idx = INDEX(x, y, NX);
                const double ux_local = ux[idx];
                const double uy_local = uy[idx];
                velocity_magn_mat.at<float>(y, x) = ux_local * ux_local + uy_local * uy_local;
            }
        }

        // Normalize the matrices to 0-255 for display
        cv::normalize(velocity_magn_mat, velocity_magn_norm, 0, 255, cv::NORM_MINMAX);

        //8-bit images
        velocity_magn_norm.convertTo(velocity_magn_norm, CV_8U);

        // Apply color maps
        cv::applyColorMap(velocity_magn_norm, velocity_heatmap, cv::COLORMAP_PLASMA);

        //Flip the image vertically (OpenCV works in the opposite way than our code)
        cv::flip(velocity_heatmap, velocity_heatmap, 0); //flips along the x axis

        // Add frame to video
        video_writer.write(velocity_heatmap);
        
}
