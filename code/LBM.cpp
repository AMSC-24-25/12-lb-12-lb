#include "LBM.hpp"
#include <cmath>
#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>


// Constructor
LBmethod::LBmethod(const unsigned int NSTEPS, const unsigned int NX,const unsigned int NY,  const double u_lid, const double Re, const double rho0, const unsigned int num_cores)
    : NSTEPS(NSTEPS), NX(NX), NY(NY), u_lid(u_lid), Re(Re), rho0(rho0), num_cores(num_cores), nu((u_lid * NY) / Re), tau(3.0 * nu + 0.5),
      direction({std::make_pair(0, 0),   //Rest direction
                 std::make_pair(1, 0),   //Right
                 std::make_pair(0, 1),   //Up
                 std::make_pair(-1, 0),  //Left
                 std::make_pair(0, -1),  //Down
                 std::make_pair(1, 1),   //Top-right diagonal
                 std::make_pair(-1, 1),  //Top-left diagonal
                 std::make_pair(-1, -1), //Bottom-left diagonal
                 std::make_pair(1, -1)}),//Bottom-right diagonal
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
    rho.assign(NX * NY, rho0); // Density initialized to rho0 everywhere
    u.assign(NX * NY, {0.0, 0.0}); // Velocity initialized to 0
    f_eq.assign(NX * NY * ndirections, 0.0); // Equilibrium distribution function array
    f.assign(NX * NY * ndirections, 0.0); //  Distribution function array
    f_temp.assign(NX * NY * ndirections, 0.0);

    #pragma omp parallel for
    for (unsigned int x = 0; x < NX; ++x) {
        for (unsigned int y = 0; y < NY; ++y) {
            for (unsigned int i = 0; i < ndirections; ++i) {
                f[INDEX3D(x, y, i, NX, ndirections)] = weight[i];
                f_eq[INDEX3D(x, y, i, NX, ndirections)] = weight[i];
            }
        }
    }
}

void LBmethod::Equilibrium() {
    // Compute the equilibrium distribution function f_eq
        //collapse(2) combines the 2 loops into a single iteration space-> convient when I have large Nx and Ny (not when they're really different tho)
        //static ensure uniform distribution
        //I don't do collapse(3) because the inner loop is light
        #pragma omp parallel for schedule(static)
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

void LBmethod::UpdateMacro() {
    #pragma omp for schedule(static) private(rho_local, ux_local, uy_local)
        //or schedule(dynamic, chunk_size) if the computational complexity varies
        for (unsigned int x=0; x<NX; ++x){
            for (unsigned int y = 0; y < NY; ++y) {
                size_t idx = INDEX(x, y, NX);
                
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

void LBmethod::Collisions() {
    #pragma omp for schedule(static)
        //we use f=f-(f-f_eq)/tau from BGK
        for (unsigned int x=0;x<NX;++x){
            for (unsigned int y=0;y<NY;++y){
                for (unsigned int i=0;i<ndirections;++i){
                    f[INDEX3D(x, y, i, NX, ndirections)]=f[INDEX3D(x, y, i, NX, ndirections)]-(f[INDEX3D(x, y, i, NX, ndirections)]-f_eq[INDEX3D(x, y, i, NX, ndirections)])/tau;
                }
            }
        }
}

void LBmethod::Streaming() {
    //f(x,y,t+1)=f(x-cx,y-cy,t)
        std::vector<int> opposites = {0, 3, 4, 1, 2, 7, 8, 5, 6}; //Opposite velocities

        //paralleliation only in the bulk streaming
        //Avoid at boundaries to prevent race conditions
        #pragma omp for schedule(static)
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
        // Boundary conditions are applied serially to avoid race conditions
        #pragma omp single
        {
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
        }

        std::swap(f, f_temp);//f_temp is f at t=t+1 so now we use the new function f_temp in f
}

void LBmethod::Run_simulation() {
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
            
            #pragma omp parallel
            {
                Collisions();
                Streaming();
                UpdateMacro();
            }
            
            Visualization(t);
        }
}

void LBmethod::Visualization(unsigned int t) {
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
        #pragma omp parallel for
        for (unsigned int x = 0; x < NX; ++x) {
            for (unsigned int y = 0; y < NY; ++y) {
                size_t idx = INDEX(x, y, NX);
                double ux = u[idx].first;
                double uy = u[idx].second;
                velocity_magn_mat.at<float>(y, x) = ux * ux + uy * uy;
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
