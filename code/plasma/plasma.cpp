#include "plasma.hpp"
#include <cmath>
#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>


// Constructor
LBmethod::LBmethod(const size_t NSTEPS, const size_t NX,const size_t NY, const size_t num_cores, const size_t Z_ion, const size_t A_ion, const double r_ion, const double tau_ion,  const double tau_el)
    : NSTEPS(NSTEPS), NX(NX), NY(NY), num_cores(num_cores),Z_ion(Z_ion), A_ion(A_ion),r_ion(r_ion), tau_ion(tau_ion), tau_el(tau_el),
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

void LBmethod::SolvePoisson() {
    // Poisson equation in SI units: \nabla^2 phi = -rho_c / eps_0
    // phi = electric potential (V), eps_0 = vacuum permittivity (F/m)

    const double inveps=1.0/eps_0;
    const double omega=1.8; //Over relaxation factor (1=Gauss-Seidal method, 1.5-1.9=Successive over relaxation )
    const int max_iter=100000;
    const double tol = 1e-4;  // Convergence tolerance

    std::vector<double> rho_c(NX * NY, 0.0);
    static std::vector<double> rho_c_old(NX * NY, 0.0); //no need to be passed->static

    // Compute total charge density: rho_c = q_ion * rho_ion - q_el * rho_el
    #pragma omp parallel for collapse(2)
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
            const size_t idx = INDEX(x, y, NX);
            rho_c[idx] = Z_ion* (-q_el) * rho_ion[idx] - q_el * rho_el[idx];
        }
    }

    double max_change = 0.0;

    #pragma omp parallel for reduction(max:max_change)
    for (size_t idx = 0; idx < NX * NY; ++idx) {
        double delta = std::abs(rho_c[idx] - rho_c_old[idx]);
        if (delta > max_change) max_change = delta;
    }

    ///////Check for other methods to solve Poisson equation
    // Solve Poisson equation using Gauss-Seidel (/SOR)
    const double threshold=10e3;

    if (max_change > threshold) {
        std::cout << "Large change in rho_c detected (Δρ = " << max_change << "), resetting phi." << std::endl;
        std::fill(phi.begin(), phi.end(), 0.0);
    }

    double max_error;
    int iter = 0;

    do {
        max_error = 0.0;

        for (size_t x = 1; x < NX - 1; ++x) {
            for (size_t y = 1; y < NY - 1; ++y) {
                size_t idx = INDEX(x, y, NX);

                double phi_old = phi[idx];

                double phi_new_local = 0.25 * (
                    phi[INDEX(x + 1, y, NX)] +
                    phi[INDEX(x - 1, y, NX)] +
                    phi[INDEX(x, y + 1, NX)] +
                    phi[INDEX(x, y - 1, NX)] -
                    rho_c[idx] * inveps
                );

                // Gauss-Seidel (+overrelaxation)
                phi[idx] = phi_old + omega * (phi_new_local - phi_old);
                max_error = std::max(max_error, std::abs(phi[idx] - phi_old));
            }
        }

        // Dirichlet Boundary Conditions: φ = 0
        for (size_t x = 0; x < NX; ++x) {
            phi[INDEX(x, 0, NX)] = 0.0;
            phi[INDEX(x, NY - 1, NX)] = 0.0;
        }
        for (size_t y = 0; y < NY; ++y) {
            phi[INDEX(0, y, NX)] = 0.0;
            phi[INDEX(NX - 1, y, NX)] = 0.0;
        }

        ++iter;

        if (iter % 100 == 0) {
            std::cout << "Poisson iter " << iter << ", max_error = " << max_error << std::endl;
        }

    } while (max_error > tol && iter < max_iter);

    if (iter >= max_iter) {
        std::cerr << "Warning: Poisson solver did not converge after " << max_iter << " iterations. Max error: " << max_error << std::endl;
    }

    // Verify charge density conservation for debug
    double total_charge = 0.0;
    for (size_t idx = 0; idx < NX * NY; ++idx){
         total_charge += rho_c[idx];
    }
       
    std::cout << "Total net charge in domain: " << total_charge << " C" << std::endl;
    std::copy(rho_c.begin(), rho_c.end(), rho_c_old.begin());


}

void LBmethod::SolvePoisson_fft() {
    //Periodic conditions ONLY

    const double inveps = 1.0 / eps_0;

    // Allocate FFTW arrays
    fftw_complex *rho_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NX * NY);
    fftw_complex *phi_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NX * NY);

    fftw_plan forward_plan = fftw_plan_dft_r2c_2d(NX, NY, phi.data(), rho_hat, FFTW_ESTIMATE);
    fftw_plan backward_plan = fftw_plan_dft_c2r_2d(NX, NY, phi_hat, phi.data(), FFTW_ESTIMATE);

    std::vector<double> rho_c(NX * NY, 0.0);

    // Compute charge density: rho_c = Z_ion * q_ion * rho_ion - q_el * rho_el
    #pragma omp parallel for collapse(2)
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
            size_t idx = INDEX(x, y, NX);
            rho_c[idx] = Z_ion * (-q_el) * rho_ion[idx] + q_el * rho_el[idx]; //Qua ho messo un + perché qel è già negativa
        }
    }

    // Copy rho_c into phi temporarily to use r2c FFT (input must be real)
    std::copy(rho_c.begin(), rho_c.end(), phi.begin());

    // FFT forward: rho_hat = FFT[rho_c]
    fftw_execute_dft_r2c(forward_plan, phi.data(), rho_hat);

    // Solve in Fourier space
    for (size_t i = 0; i < NX; ++i) {
        int kx = (i <= NX/2) ? i : (int)i - NX;
        double kx2 = 4.0 * std::pow(std::sin(M_PI * kx / NX), 2);

        for (size_t j = 0; j < NY; ++j) {
            int ky = (j <= NY/2) ? j : (int)j - NY;
            double ky2 = 4.0 * std::pow(std::sin(M_PI * ky / NY), 2);

            double denom = (kx2 + ky2);
            size_t idx = i * NY + j;

            if (denom != 0.0) {
                double scale = -1.0 / (inveps * denom);
                phi_hat[idx][0] = rho_hat[idx][0] * scale;
                phi_hat[idx][1] = rho_hat[idx][1] * scale;
            } else {
                phi_hat[idx][0] = 0.0; // remove DC offset (zero total potential)
                phi_hat[idx][1] = 0.0;
            }
        }
    }

    // Inverse FFT: phi = IFFT[phi_hat]
    fftw_execute_dft_c2r(backward_plan, phi_hat, phi.data());

    // Normalize FFT result (FFTW doesn't normalize)
    double norm_factor = 1.0 / (NX * NY);
    #pragma omp parallel for
    for (size_t idx = 0; idx < NX * NY; ++idx) {
        phi[idx] *= norm_factor;
    }

    // Clean up
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);
    fftw_free(rho_hat);
    fftw_free(phi_hat);

    // Debug: check total charge (should be conserved)
    double total_charge = 0.0;
    for (size_t idx = 0; idx < NX * NY; ++idx) {
        total_charge += rho_c[idx];
    }
    std::cout << "Total net charge in domain: " << total_charge << " C" << std::endl;
}


void LBmethod::Collisions() {
    // Compute electric field: E = -∇φ + E_ext

    std::vector<double> Ex(NX * NY, 0.0), Ey(NX * NY, 0.0);

    #pragma omp parallel for collapse(2)
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
            size_t idx = INDEX(x, y, NX);

            // Apply periodic boundary conditions with modular arithmetic
            size_t xp1 = (x + 1) % NX;
            size_t xm1 = (x + NX - 1) % NX;
            size_t yp1 = (y + 1) % NY;
            size_t ym1 = (y + NY - 1) % NY;

            Ex[idx] = -(phi[INDEX(xp1, y, NX)] - phi[INDEX(xm1, y, NX)]) / 2.0 + Ex_ext;
            Ey[idx] = -(phi[INDEX(x, yp1, NX)] - phi[INDEX(x, ym1, NX)]) / 2.0 + Ey_ext;
        }
    }

    // Collision step with Guo forcing term
    #pragma omp parallel for collapse(2)
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
            const size_t idx = INDEX(x, y, NX);

            const double Ex_loc = Ex[idx];
            const double Ey_loc = Ey[idx];

            for (size_t i = 0; i < ndirections; ++i) {
                const double F_ion = weight[i] * rho_ion[idx] * (1.0 - 1.0 / (2.0 * tau_ion)) * (
                    3.0 * ((directionx[i] - ux_ion[idx]) * Z_ion * (-q_el) * Ex_loc / (A_ion * m_p) +
                           (directiony[i] - uy_ion[idx]) * Z_ion * (-q_el) * Ey_loc / (A_ion * m_p)) +
                    9.0 * (directionx[i] * ux_ion[idx] * directionx[i] * Z_ion * (-q_el) * Ex_loc / (A_ion * m_p) +
                           directiony[i] * uy_ion[idx] * directiony[i] * Z_ion * (-q_el) * Ey_loc / (A_ion * m_p))
                );

                const double F_el = weight[i] * rho_el[idx] * (1.0 - 1.0 / (2.0 * tau_el)) * (
                    3.0 * ((directionx[i] - ux_el[idx]) * (q_el) * Ex_loc / (m_el) +
                           (directiony[i] - uy_el[idx]) * (q_el) * Ey_loc / (m_el)) +
                    9.0 * (directionx[i] * ux_el[idx] * directionx[i] * (q_el) * Ex_loc / (m_el) +
                           directiony[i] * uy_el[idx] * directiony[i] * (q_el) * Ey_loc / (m_el))
                );

                const size_t fi_idx = INDEX(x, y, i, NX, ndirections);
                f_ion[fi_idx] += -(f_ion[fi_idx] - f_eq_ion[fi_idx]) / tau_ion + F_ion;
                f_el[fi_idx]  += -(f_el[fi_idx]  - f_eq_el[fi_idx]) / tau_el + F_el;
            }
        }
    }
}

void LBmethod::Streaming_periodic() {
    // f(x,y,t+1) = f(x-cx, y-cy, t) con condizioni periodiche
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
            for (size_t i = 0; i < ndirections; ++i) {
                // Origin coordinates (periodic wrapping)
                size_t x_src = (x + NX - directionx[i]) % NX;
                size_t y_src = (y + NY - directiony[i]) % NY;

                f_temp_ion[INDEX(x, y, i, NX, ndirections)] = f_ion[INDEX(x_src, y_src, i, NX, ndirections)];
                f_temp_el[INDEX(x, y, i, NX, ndirections)]  = f_el[INDEX(x_src, y_src, i, NX, ndirections)];
            }
        }
    }

    f_ion.swap(f_temp_ion);
    f_el.swap(f_temp_el);
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
        //setting envirolement constant to plot properly (can e included in the function below if sure that it is ok)
        const int border = 10, label_height = 30;
        const int tile_w = NX + 2 * border;
        const int tile_h = NY + 2 * border + label_height;
        const int frame_w = 3 * tile_w + 20;  // 3 tiles + legend (20px)
        const int frame_h = 2 * tile_h;
        // VideoWriter setup
        const std::string video_filename = "simulation_plasma.mp4";
        const double fps = 10.0; // Frames per second for the video
        video_writer.open(video_filename, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(frame_w, frame_h));


        if (!video_writer.isOpened()) {
            std::cerr << "Error: Could not open the video writer." << std::endl;
            return;
        }

        SolvePoisson_fft(); 
        UpdateMacro();

        for (size_t t=0; t<NSTEPS; ++t){
            Collisions();
            Streaming_periodic();
            UpdateMacro();
            SolvePoisson_fft();
            Visualization(t);

            // nabla^2 phi = -rho_c / eps_0 
            //SolvePoisson();
            //std::cout<<"Ok Poisson step "<<t<<std::endl;
            //Collisions();// f(x,y,t+1)=f(x-cx,y-cy,t) + tau * (f_eq - f) + dt*F
            //std::cout<<"Ok Collisions step "<<t<<std::endl;
            //Streaming_periodic();// f(x,y,t+1)=f(x-cx,y-cy,t)
            //std::cout<<"Ok Streaming step "<<t<<std::endl;
            //UpdateMacro();// rho=sum(f), ux=sum(f*c_x)/rho, uy=sum(f*c_y)/rho
            //std::cout<<"Ok UpdateMacro step "<<t<<std::endl;
            //Visualization(t);
            //std::cout<<"Ok Visualization step "<<t<<std::endl;
        }
    
        video_writer.release();
        std::cout << "Video saved as " << video_filename << std::endl;
}

void LBmethod::Visualization(size_t t) {
    constexpr int border = 10;
    constexpr int label_height = 30;

    static cv::Mat velocity_magn_mat_ion, velocity_magn_mat_el;
    static cv::Mat velocity_heatmap_ion, velocity_heatmap_el;
    static cv::Mat density_mat_ion, density_mat_el;
    static cv::Mat density_heatmap_ion, density_heatmap_el;
    static cv::Mat combined_density_mat, combined_velocity_mat;
    static cv::Mat combined_density_heatmap, combined_velocity_heatmap;
    static cv::Mat output_frame;

    if (t == 0) {
        velocity_magn_mat_ion = cv::Mat(NY, NX, CV_32F);
        velocity_magn_mat_el = cv::Mat(NY, NX, CV_32F);
        density_mat_ion = cv::Mat(NY, NX, CV_32F);
        density_mat_el = cv::Mat(NY, NX, CV_32F);
        combined_density_mat = cv::Mat(NY, NX, CV_32F);
        combined_velocity_mat = cv::Mat(NY, NX, CV_32F);
    }

    // Fill raw data matrices
    #pragma omp parallel for collapse(2)
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
            const size_t idx = INDEX(x, y, NX);

            const double ux_i = ux_ion[idx], uy_i = uy_ion[idx];
            const double ux_e = ux_el[idx], uy_e = uy_el[idx];
            const double rho_i = rho_ion[idx], rho_e = rho_el[idx];

            velocity_magn_mat_ion.at<float>(y, x) = std::hypot(ux_i, uy_i);
            velocity_magn_mat_el.at<float>(y, x) = std::hypot(ux_e, uy_e);
            density_mat_ion.at<float>(y, x) = static_cast<float>(rho_i);
            density_mat_el.at<float>(y, x) = static_cast<float>(rho_e);

            combined_density_mat.at<float>(y, x) = static_cast<float>(rho_i - rho_e);
            combined_velocity_mat.at<float>(y, x) = std::hypot(ux_i - ux_e, uy_i - uy_e);
        }
    }

    // Apply color maps
    apply_colormap(velocity_magn_mat_ion, velocity_heatmap_ion, cv::COLORMAP_PLASMA);
    apply_colormap(velocity_magn_mat_el, velocity_heatmap_el, cv::COLORMAP_PLASMA);
    apply_colormap(density_mat_ion, density_heatmap_ion, cv::COLORMAP_JET);
    apply_colormap(density_mat_el, density_heatmap_el, cv::COLORMAP_JET);
    apply_colormap(combined_density_mat, combined_density_heatmap, cv::COLORMAP_JET);
    apply_colormap(combined_velocity_mat, combined_velocity_heatmap, cv::COLORMAP_PLASMA);

    // Flip vertically
    auto flipv = [](cv::Mat& m) { cv::flip(m, m, 0); };
    flipv(velocity_heatmap_ion);
    flipv(velocity_heatmap_el);
    flipv(density_heatmap_ion);
    flipv(density_heatmap_el);
    flipv(combined_density_heatmap);
    flipv(combined_velocity_heatmap);

    // Helper: wrap with border and label
    auto wrap_with_label = [&](const cv::Mat& src, const std::string& label) -> cv::Mat {
        cv::Mat bordered;
        cv::copyMakeBorder(src, bordered, border, border + label_height, border, border, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
        cv::putText(bordered, label, cv::Point(border + 5, bordered.rows - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        return bordered;
    };

    // Compose labeled frames
    cv::Mat top_row, bottom_row;
    cv::hconcat(std::vector<cv::Mat>{wrap_with_label(density_heatmap_ion, "rho Ions"), wrap_with_label(density_heatmap_el, "rho Electrons"), wrap_with_label(combined_density_heatmap, "Comb rho")}, top_row);
    cv::hconcat(std::vector<cv::Mat>{wrap_with_label(velocity_heatmap_ion, "v Ions"), wrap_with_label(velocity_heatmap_el, "v Electrons"), wrap_with_label(combined_velocity_heatmap, "Comb v")}, bottom_row);

    // Compose full visualization
    cv::Mat grid;
    cv::vconcat(std::vector<cv::Mat>{top_row, bottom_row}, grid);

    // Add legend
    cv::Mat legend(200, 20, CV_8UC3);  // tall vertical legend
    for (int i = 0; i < legend.rows; ++i) {
        legend.row(i).setTo(cv::Vec3b(255 * i / legend.rows, 255 * i / legend.rows, 255 * i / legend.rows));
    }
    cv::applyColorMap(legend, legend, cv::COLORMAP_JET);
    cv::resize(legend, legend, cv::Size(20, grid.rows));  // same height as output

    // Final layout: [ grid | legend ]
    cv::hconcat(std::vector<cv::Mat>{grid, legend}, output_frame);

    // Write to video
    video_writer.write(output_frame);
}
