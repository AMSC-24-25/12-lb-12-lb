#include "plasma.hpp"
#include <cmath>
#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>


// Constructor
LBmethod::LBmethod(const size_t NSTEPS, const size_t NX,const size_t NY, const double Re, const size_t num_cores, const size_t tau_ion,  const size_t tau_el)
    : NSTEPS(NSTEPS), NX(NX), NY(NY), Re(Re), num_cores(num_cores), tau_ion(tau_ion), tau_el(tau_el),
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
    //IONS
    rho_ion.assign(NX * NY, 1.0); // Density initialized to 1 everywhere
    ux_ion.assign(NX * NY, 0.0); // Velocity initialized to 0
    uy_ion.assign(NX * NY, 0.0); // Velocity initialized to 0
    f_eq_ion.assign(NX * NY * ndirections, 0.0); // Equilibrium distribution function array
    f_ion.assign(NX * NY * ndirections, 0.0); //  Distribution function array
    f_temp_ion.assign(NX * NY * ndirections, 0.0);
    //ELECTRONS
    rho_el.assign(NX * NY, 1.0); // Density initialized to 1 everywhere
    ux_el.assign(NX * NY, 0.0); // Velocity initialized to 0
    uy_el.assign(NX * NY, 0.0); // Velocity initialized to 0
    f_eq_el.assign(NX * NY * ndirections, 0.0); // Equilibrium distribution function array
    f_el.assign(NX * NY * ndirections, 0.0); //  Distribution function array
    f_temp_el.assign(NX * NY * ndirections, 0.0);
    //Poisson equation:
    phi.assign(NX * NY, 0.0);
    phi_new.assign(NX * NY, 0.0);
    
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < NX * NY * ndirections; ++idx) {
        const size_t i = idx % ndirections; // Use size_t for consistency
        f_ion[idx] = f_eq_ion[idx] = weight[i];
        f_el[idx] = f_eq_el[idx] = weight[i];
    }

}

void LBmethod::Equilibrium() {
    // Compute the equilibrium distribution function f_eq
    #pragma omp parallel for collapse(2) schedule(static)
        for (size_t x = 0; x < NX; ++x) {
            for (size_t y = 0; y < NY; ++y) {
                const size_t idx = INDEX(x, y, NX); // Get 1D index for 2D point (x, y)
                const double u2_ion = ux_ion[idx] * ux_ion[idx] + uy_ion[idx] * uy_ion[idx]; // Square of the speed magnitude
                const double u2_el = ux_el[idx] * ux_el[idx] + uy_el[idx] * uy_el[idx];

                for (size_t i = 0; i < ndirections; ++i) {
                    const double cu_ion = (directionx[i] * ux_ion[idx] + directiony[i] * uy_ion[idx]); // Dot product (c_i · u)
                    const double cu_el = (directionx[i] * ux_el[idx] + directiony[i] * uy_el[idx]); 

                    // Compute f_eq using the BGK collision formula
                    f_eq_ion[INDEX(x, y, i, NX, ndirections)] = weight[i] * rho_ion[idx] * (1.0 + 3.0 * cu_ion + 4.5 * cu_ion * cu_ion - 1.5 * u2_ion);
                    f_eq_el[INDEX(x, y, i, NX, ndirections)] = weight[i] * rho_el[idx] * (1.0 + 3.0 * cu_el + 4.5 * cu_el * cu_el - 1.5 * u2_el);
                }
            }
        }
}

void LBmethod::UpdateMacro() {
    double rho_local_ion = 0.0;
    double ux_local_ion = 0.0;
    double uy_local_ion = 0.0;

    double rho_local_el = 0.0;
    double ux_local_el = 0.0;
    double uy_local_el = 0.0;

    #pragma omp parallel for collapse(2) private(rho_local_ion, ux_local_ion, uy_local_ion, rho_local_el, ux_local_el, uy_local_el)
        for (size_t x=0; x<NX; ++x){
            for (size_t y = 0; y < NY; ++y) {
                const size_t idx = INDEX(x, y, NX);
                rho_local_ion = 0.0;
                ux_local_ion = 0.0;
                uy_local_ion = 0.0;

                rho_local_el = 0.0;
                ux_local_el = 0.0;
                uy_local_el = 0.0;


                for (size_t i = 0; i < ndirections; ++i) {
                    const double fi_ion=f_ion[INDEX(x, y, i, NX, ndirections)];
                    rho_local_ion += fi_ion;
                    ux_local_ion += fi_ion * directionx[i];
                    uy_local_ion += fi_ion * directiony[i];

                    const double fi_el=f_el[INDEX(x, y, i, NX, ndirections)];
                    rho_local_el += fi_el;
                    ux_local_el += fi_el * directionx[i];
                    uy_local_el += fi_el * directiony[i];
                    
                }
                if (rho_local_ion<1e-10){
                    rho_ion[idx] = 0.0;
                    ux_ion[idx] = 0.0;
                    uy_ion[idx] = 0.0;
                }else if (rho_local_el<1e-10){
                    rho_el[idx] = 0.0;
                    ux_el[idx] = 0.0;
                    uy_el[idx] = 0.0;
                }
                else {
                    rho_ion[idx] = rho_local_ion;
                    ux_ion[idx]=ux_local_ion/rho_local_ion;
                    uy_ion[idx]=uy_local_ion/rho_local_ion;

                    rho_el[idx] = rho_local_el;
                    ux_el[idx]=ux_local_el/rho_local_el;
                    uy_el[idx]=uy_local_el/rho_local_el;
                }
            }
        }
        Equilibrium();
}

void LBmethod::SolvePoisson(){
    //Poisson equation: \nablasqr phi = -rho_c/epsilon
    //phi=electric potential
    // Compute charge density
    std::vector<double> rho_c(NX * NY, 0.0);

    #pragma omp parallel for collapse(2)
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
            const size_t idx = INDEX(x, y, NX);
            rho_c[idx] = q_ion * rho_ion[idx] - q_el * rho_el[idx];  // Charge density
        }
    }

    // Solve Poisson equation (for simplicity, using Jacobi iteration)
    double tol = 1e-6;  // Convergence tolerance
    double max_error;
    do {
        max_error = 0.0;
        
        for (size_t x = 1; x < NX - 1; ++x) {
            for (size_t y = 1; y < NY - 1; ++y) {
                const size_t idx = INDEX(x, y, NX);
                phi_new[idx] = 0.25 * (phi[INDEX(x + 1, y, NX)] + 
                                       phi[INDEX(x - 1, y, NX)] + 
                                       phi[INDEX(x, y + 1, NX)] + 
                                       phi[INDEX(x, y - 1, NX)] - 
                                       rho_c[idx]);
                max_error = std::max(max_error, std::abs(phi_new[idx] - phi[idx]));
            }
        }
        phi = phi_new;  // Update the potential
        phi.swap(phi_new);
    } while (max_error > tol);
    this->phi =phi; //store the result
}

void LBmethod::Collisions() {
    //Compute electric field from phi gradient: 
    std::vector<double> Ex(NX * NY, 0.0), Ey(NX * NY, 0.0);
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t x = 1; x < NX - 1; ++x) {
        for (size_t y = 1; y < NY - 1; ++y) {
            const size_t idx = INDEX(x, y, NX);
            Ex[idx] = -(phi[INDEX(x + 1, y, NX)] - phi[INDEX(x - 1, y, NX)]) / 2.0;
            Ey[idx] = -(phi[INDEX(x, y + 1, NX)] - phi[INDEX(x, y - 1, NX)]) / 2.0;
        }
    }

    //Apply forces unew=u + tau*F
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
            const size_t idx = INDEX(x, y, NX);

            // Apply electric force for ions (F=qa*E)
            
            if (rho_ion[idx] > 1e-10) {
                ux_ion[idx] += tau_ion * q_ion * Ex[idx] / rho_ion[idx];
                uy_ion[idx] += tau_ion * q_ion * Ey[idx] / rho_ion[idx];
            }
            if (rho_el[idx] > 1e-10) {
                ux_el[idx] -= tau_el * q_el * Ex[idx] / rho_el[idx];
                uy_el[idx] -= tau_el * q_el * Ey[idx] / rho_el[idx];
            }
        }
    }
    //Collision step (BGK)
    #pragma omp parallel for collapse(3)
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
            for (size_t i = 0; i < ndirections; ++i) {
                const size_t idx = INDEX(x, y, i, NX, ndirections);
                f_ion[idx] -= (f_ion[idx] - f_eq_ion[idx]) / tau_ion;
                f_el[idx] -= (f_el[idx] - f_eq_el[idx]) / tau_el;
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
                        f_temp_ion[INDEX(x, y, i, NX, ndirections)] = f_ion[INDEX(x_str, y_str, i, NX, ndirections)];
                        f_temp_el[INDEX(x, y, i, NX, ndirections)] = f_el[INDEX(x_str, y_str, i, NX, ndirections)];
                    }
                }
            }
        }
        // Parallelize the boundary conditions
        // Sides + bottom angles
        #pragma omp for schedule(static)
        for (size_t y = 0; y < NY; ++y) {
            // Left
            f_temp_ion[INDEX(0, y, 1, NX, ndirections)] = f_ion[INDEX(0, y, 3, NX, ndirections)];
            f_temp_ion[INDEX(0, y, 8, NX, ndirections)] = f_ion[INDEX(0, y, 6, NX, ndirections)];
            f_temp_ion[INDEX(0, y, 5, NX, ndirections)] = f_ion[INDEX(0, y, 7, NX, ndirections)];
            // Right
            f_temp_ion[INDEX(NX - 1, y, 3, NX, ndirections)] = f_ion[INDEX(NX - 1, y, 1, NX, ndirections)];
            f_temp_ion[INDEX(NX - 1, y, 7, NX, ndirections)] = f_ion[INDEX(NX - 1, y, 5, NX, ndirections)];
            f_temp_ion[INDEX(NX - 1, y, 6, NX, ndirections)] = f_ion[INDEX(NX - 1, y, 8, NX, ndirections)];

            f_temp_el[INDEX(0, y, 1, NX, ndirections)] = f_el[INDEX(0, y, 3, NX, ndirections)];
            f_temp_el[INDEX(0, y, 8, NX, ndirections)] = f_el[INDEX(0, y, 6, NX, ndirections)];
            f_temp_el[INDEX(0, y, 5, NX, ndirections)] = f_el[INDEX(0, y, 7, NX, ndirections)];
            // Right
            f_temp_el[INDEX(NX - 1, y, 3, NX, ndirections)] = f_el[INDEX(NX - 1, y, 1, NX, ndirections)];
            f_temp_el[INDEX(NX - 1, y, 7, NX, ndirections)] = f_el[INDEX(NX - 1, y, 5, NX, ndirections)];
            f_temp_el[INDEX(NX - 1, y, 6, NX, ndirections)] = f_el[INDEX(NX - 1, y, 8, NX, ndirections)];
        }

        #pragma omp for schedule(static)
        for (size_t x = 0; x < NX; ++x) {
            // Bottom boundary conditions
            f_temp_ion[INDEX(x, 0, 2, NX, ndirections)] = f_ion[INDEX(x, 0, 4, NX, ndirections)];
            f_temp_ion[INDEX(x, 0, 5, NX, ndirections)] = f_ion[INDEX(x, 0, 7, NX, ndirections)];
            f_temp_ion[INDEX(x, 0, 6, NX, ndirections)] = f_ion[INDEX(x, 0, 8, NX, ndirections)];

            f_temp_el[INDEX(x, 0, 2, NX, ndirections)] = f_el[INDEX(x, 0, 4, NX, ndirections)];
            f_temp_el[INDEX(x, 0, 5, NX, ndirections)] = f_el[INDEX(x, 0, 7, NX, ndirections)];
            f_temp_el[INDEX(x, 0, 6, NX, ndirections)] = f_el[INDEX(x, 0, 8, NX, ndirections)];

            // Top boundary conditions
            f_temp_ion[INDEX(x, NY - 1, 4, NX, ndirections)] = f_ion[INDEX(x, NY - 1, 2, NX, ndirections)];
            f_temp_ion[INDEX(x, NY - 1, 7, NX, ndirections)] = f_ion[INDEX(x, NY - 1, 5, NX, ndirections)];
            f_temp_ion[INDEX(x, NY - 1, 8, NX, ndirections)] = f_ion[INDEX(x, NY - 1, 6, NX, ndirections)];

            f_temp_el[INDEX(x, NY - 1, 4, NX, ndirections)] = f_el[INDEX(x, NY - 1, 2, NX, ndirections)];
            f_temp_el[INDEX(x, NY - 1, 7, NX, ndirections)] = f_el[INDEX(x, NY - 1, 5, NX, ndirections)];
            f_temp_el[INDEX(x, NY - 1, 8, NX, ndirections)] = f_el[INDEX(x, NY - 1, 6, NX, ndirections)];
        }
    }
    f_ion.swap(f_temp_ion);
    f_el.swap(f_temp_el);
    //f_temp is f at t=t+1 so now we use the new function f_temp in f
}

void LBmethod::Run_simulation() {
    
    // Set threads for this simulation
        omp_set_num_threads(num_cores);
        
        // VideoWriter setup
        const std::string video_filename = "simulation_plasma.mp4";
        const double fps = 10.0; // Frames per second for the video
        video_writer.open(video_filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(NX, NY));

        if (!video_writer.isOpened()) {
            std::cerr << "Error: Could not open the video writer." << std::endl;
            return;
        }
    
        for (size_t t=0; t<NSTEPS; ++t){
            const double t_double = static_cast<double>(t);//avoid repeated type cast
            
            SolvePoisson();
            std::cout<<"ok poisson"<<std::endl;
            Collisions();
            std::cout<<"ok collision"<<std::endl;
            Streaming();
            std::cout<<"ok streaming"<<std::endl;
            UpdateMacro();
            std::cout<<"ok update"<<std::endl;
            Visualization(t);
        }
    
        video_writer.release();
        std::cout << "Video saved as " << video_filename << std::endl;
}

void LBmethod::Visualization(size_t t) {
        static cv::Mat velocity_magn_mat_ion;
        static cv::Mat velocity_magn_mat_el;
        static cv::Mat velocity_heatmap_ion;
        static cv::Mat velocity_heatmap_el;

        static cv::Mat density_mat_ion;
        static cv::Mat density_mat_el;
        static cv::Mat density_heatmap_ion;
        static cv::Mat density_heatmap_el;

        static cv::Mat combined_density_mat;
        static cv::Mat combined_velocity_mat;
        static cv::Mat combined_density_heatmap;
        static cv::Mat combined_velocity_heatmap;

        static cv::Mat output_frame;


        // Initialize only when t == 0
        if (t == 0) {
        // Initialize the heatmaps with the same size as the grid
            //OpenCV uses a row-major indexing
            velocity_magn_mat_ion = cv::Mat(NY, NX, CV_32F);
            velocity_magn_mat_el = cv::Mat(NY, NX, CV_32F);

            density_mat_ion = cv::Mat(NY, NX, CV_32F);
            density_mat_el = cv::Mat(NY, NX, CV_32F);
    

            // Create heatmap images (8 bit images)
            velocity_heatmap_ion = cv::Mat(NY, NX, CV_8UC3);
            velocity_heatmap_el = cv::Mat(NY, NX, CV_8UC3);

            density_heatmap_ion = cv::Mat(NY, NX, CV_8UC3);
            density_heatmap_el = cv::Mat(NY, NX, CV_8UC3);

            combined_density_mat = cv::Mat(NY, NX, CV_32F);
            combined_velocity_mat = cv::Mat(NY, NX, CV_32F);

            combined_density_heatmap = cv::Mat(NY, NX, CV_8UC3);
            combined_velocity_heatmap = cv::Mat(NY, NX, CV_8UC3);

            output_frame = cv::Mat(2 * NY, 3 * NX, CV_8UC3);
        }

        // Fill matrices with new data
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t x = 0; x < NX; ++x) {
            for (size_t y = 0; y < NY; ++y) {
                const size_t idx = INDEX(x, y, NX);

                const double ux_local_ion = ux_ion[idx];
                const double uy_local_ion = uy_ion[idx];
                velocity_magn_mat_ion.at<float>(y, x) = std::sqrt(ux_local_ion * ux_local_ion + uy_local_ion * uy_local_ion);

                const double ux_local_el = ux_el[idx];
                const double uy_local_el = uy_el[idx];
                velocity_magn_mat_el.at<float>(y, x) = std::sqrt(ux_local_el * ux_local_el + uy_local_el * uy_local_el);

                const double rho_local_ion = rho_ion[idx];
                const double rho_local_el = rho_el[idx];
                density_mat_ion.at<float>(y,x) =rho_local_ion;
                density_mat_el.at<float>(y,x) =rho_local_el;

                const double rho_diff= rho_local_ion - rho_local_el;
                const double ux_diff= ux_local_ion - ux_local_el;
                const double uy_diff= uy_local_ion - uy_local_el;
                combined_density_mat.at<float>(y,x) = rho_diff; 
                combined_velocity_mat.at<float>(y,x) = std::sqrt(ux_diff * ux_diff + uy_diff * uy_diff);


            }
        }

        // Normalize the matrices to 0-255 for display
        cv::normalize(velocity_magn_mat_ion, velocity_magn_mat_ion, 0, 255, cv::NORM_MINMAX);
        cv::normalize(velocity_magn_mat_el, velocity_magn_mat_el, 0, 255, cv::NORM_MINMAX);
        cv::normalize(density_mat_ion, density_mat_ion, 0, 255, cv::NORM_MINMAX);
        cv::normalize(density_mat_el, density_mat_el, 0, 255, cv::NORM_MINMAX);
        cv::normalize(combined_density_mat, combined_density_mat, 0, 255, cv::NORM_MINMAX);
        cv::normalize(combined_velocity_mat, combined_velocity_mat, 0, 255, cv::NORM_MINMAX);

        //8-bit images
        velocity_magn_mat_ion.convertTo(velocity_magn_mat_ion, CV_8U);
        velocity_magn_mat_el.convertTo(velocity_magn_mat_el, CV_8U);
        density_mat_ion.convertTo(density_mat_ion, CV_8U);
        density_mat_el.convertTo(density_mat_el, CV_8U);
        combined_density_mat.convertTo(combined_density_mat, CV_8U);
        combined_velocity_mat.convertTo(combined_velocity_mat, CV_8U);

        // Apply color maps
        cv::applyColorMap(velocity_magn_mat_ion, velocity_heatmap_ion, cv::COLORMAP_PLASMA);
        cv::applyColorMap(velocity_magn_mat_el, velocity_heatmap_el, cv::COLORMAP_PLASMA);
        cv::applyColorMap(density_mat_ion, density_heatmap_ion, cv::COLORMAP_JET);
        cv::applyColorMap(density_mat_el, density_heatmap_el, cv::COLORMAP_JET);

        cv::applyColorMap(combined_density_mat, combined_density_heatmap, cv::COLORMAP_JET);
        cv::applyColorMap(combined_velocity_mat, combined_velocity_heatmap, cv::COLORMAP_PLASMA);

        //Flip the image vertically (OpenCV works in the opposite way than our code)
        cv::flip(velocity_heatmap_ion, velocity_heatmap_ion, 0); //flips along the x axis
        cv::flip(velocity_heatmap_el, velocity_heatmap_el, 0);
        cv::flip(density_heatmap_ion, density_heatmap_ion, 0);
        cv::flip(density_heatmap_el, density_heatmap_el, 0);
        cv::flip(combined_density_heatmap, combined_density_heatmap, 0);
        cv::flip(combined_velocity_heatmap, combined_velocity_heatmap, 0);

        cv::Mat top_row, bottom_row;
        std::cout<<"ok";
        cv::hconcat(std::vector<cv::Mat>{density_heatmap_ion, density_heatmap_el, combined_density_heatmap}, top_row);
        cv::hconcat(std::vector<cv::Mat>{velocity_heatmap_ion, velocity_heatmap_el, combined_velocity_heatmap}, bottom_row);
        cv::vconcat(std::vector<cv::Mat>{top_row, bottom_row}, output_frame(cv::Rect(0, 60, 3 * NX, 2 * NY)));

        std::cout<<"ok";
        // Add titles above each column
        cv::putText(output_frame, "Density of Ions", cv::Point(NX / 4, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
        cv::putText(output_frame, "Density of Electrons", cv::Point(3 * NX / 4, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
        cv::putText(output_frame, "Combined Density", cv::Point(5 * NX / 4, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
        cv::putText(output_frame, "Velocity of Ions", cv::Point(NX / 4, NY + 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
        cv::putText(output_frame, "Velocity of Electrons", cv::Point(3 * NX / 4, NY + 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
        cv::putText(output_frame, "Combined Velocity", cv::Point(5 * NX / 4, NY + 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);

        // Add a legend for the color maps
        cv::Mat legend(20, 256, CV_8UC3);
        for (int i = 0; i < 256; ++i) {
            legend.col(i).setTo(cv::Vec3b(i, i, i)); // Grayscale gradient
        }
        cv::applyColorMap(legend, legend, cv::COLORMAP_JET); // Apply the same colormap
        legend.copyTo(output_frame(cv::Rect(3 * NX + 20, 60, 256, 20)));

        // Add frame to video
        video_writer.write(output_frame);
        
}
