#include "plasma.hpp"
#include <cmath>
#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

//──────────────────────────────────────────────────────────────────────────────
//  Static member definitions for D2Q9:
//──────────────────────────────────────────────────────────────────────────────
const std::array<int, LBmethod::Q> LBmethod::cx = { { 0, 1, 0, -1,  0,  1, -1, -1,  1 } };
const std::array<int, LBmethod::Q> LBmethod::cy = { { 0, 0, 1,  0, -1,  1,  1, -1, -1 } };
const std::array<double, LBmethod::Q> LBmethod::w = { {
    4.0/9.0,
    1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0
} };
// Opposite directions for bounce‐back:
const std::array<int, LBmethod::Q> LBmethod::opp = { {0,3,4,1,2,7,8,5,6} };

//──────────────────────────────────────────────────────────────────────────────
//  Constructor: everything passed in SI → convert to lattice units
//──────────────────────────────────────────────────────────────────────────────
LBmethod::LBmethod(const size_t    _NSTEPS,
                   const size_t    _NX,
                   const size_t    _NY,
                   const size_t    _n_cores,
                   const size_t    _Z_ion,
                   const size_t    _A_ion,
                   const double    _r_ion,
                   const double    _Lx_SI,
                   const double    _Ly_SI,
                   const double    _dt_SI,
                   const double    _T_e_SI,
                   const double    _T_i_SI,
                   const double    _Ex_SI,
                   const double    _Ey_SI,
                   const PoissonType _poisson_type,
                   const BCType      _bc_type,
                   const double    _omega_sor)
    : NSTEPS      (_NSTEPS),
      NX          (_NX),
      NY          (_NY),
      n_cores     (_n_cores),
      Z_ion       (_Z_ion),
      A_ion       (_A_ion),
      r_ion       (_r_ion),
      Lx_SI       (_Lx_SI),
      Ly_SI       (_Ly_SI),
      dt_SI       (_dt_SI),
      T_e_SI      (_T_e_SI),
      T_i_SI      (_T_i_SI),
      Ex_SI       (_Ex_SI),
      Ey_SI       (_Ey_SI),
      poisson_type(_poisson_type),
      bc_type     (_bc_type),
      omega_sor   (_omega_sor)
{
    // 1) Compute dx_SI, dy_SI
    dx_SI = Lx_SI / static_cast<double>(NX);
    dy_SI = Ly_SI / static_cast<double>(NY);

    // 2) In lattice units, we set dx_latt = 1, dy_latt = 1, dt_latt = 1
    dx_latt = 1.0;
    dy_latt = 1.0;
    dt_latt = 1.0;

    // 3) dt_dx = dt_SI / dx_SI (dimensionless ratio → to convert velocities)
    dt_dx = dt_SI / dx_SI;

    // 4) Compute sound speeds in lattice units:
    //      cs^2 = (k_B * T / m) * (dt_SI^2 / dx_SI^2)
    cs2_e = (kB_SI * T_e_SI / m_e_SI) * (dt_SI * dt_SI) / (dx_SI * dx_SI);
    // Ion mass in SI = A_ion * u_SI
    const double m_i_SI = static_cast<double>(A_ion) * u_SI;
    cs2_i = (kB_SI * T_i_SI / m_i_SI) * (dt_SI * dt_SI) / (dx_SI * dx_SI);
    // FIX lattice sound speed
    cs2_e = 1.0/3.0;
    cs2_i = 1.0/3.0;
    

    // 5) Compute (q/m) in lattice units:
    //    q_over_m_phys = (± e_charge_SI) / (m_species_SI)
    //    → multiply by (dt_SI^2 / dx_SI) to get lattice‐unit acceleration
    qom_e_latt = ( -e_charge_SI / m_e_SI) * (dt_SI * dt_SI) / (dx_SI);
    qom_i_latt = ( +e_charge_SI / m_i_SI) * (dt_SI * dt_SI) / (dx_SI);

    // 6) Choose relaxation times τ in lattice units.  Here we just pick two example values.
    //    If you wanted a *particular physical* viscosity ν_phys [m²/s], you could do:
    //      ν_latt = ν_phys * (dt_SI)/(dx_SI^2),
    //      τ = 0.5 + ν_latt / cs2
    //    but for now we hard‐code:
    tau_e_latt = 0.6;   // electron relaxation time
    tau_i_latt = 1.0;   // ion  relaxation time

    tau_Te_latt = 0.6;
    tau_Ti_latt = 1.0;
    kappa_e_latt = cs2_e * (tau_Te_latt - 0.5);
    kappa_i_latt = cs2_i * (tau_Ti_latt - 0.5);


    // 7) Convert the uniform SI E-field to lattice units:
    //      E_latt = E_SI * (dt_SI^2 / dx_SI)
    Ex_latt_init = Ex_SI * (dt_SI * dt_SI) / (dx_SI);
    Ey_latt_init = Ey_SI * (dt_SI * dt_SI) / (dx_SI);

    // 8) Allocate all vectors of size NX*NY and NX*NY*Q (for density, velocities, φ, etc.)
    //    Also initialize the constant fields
    //──────────────────────────────────────────────────────────────────────────────
    //  Initialize all fields at t=0:
    //    • densities = 1.0 (per cell), velocities = 0
    //    • f_e = f_e^eq(ρ=1,u=0), f_i = f_i^eq(ρ=1,u=0)
    //    • phi = 0 everywhere
    //    • Ex, Ey = Ex_latt_init, Ey_latt_init
    //    • ρ_q_phys = (ρ_i - ρ_e) * (e_charge_SI / dx_SI^3)  → initial net charge  (here 0)
    //──────────────────────────────────────────────────────────────────────────────
    f_e.assign(NX * NY * Q, 0.0);
    f_temp_e.assign(NX * NY * Q, 0.0);
    f_eq_e.assign(NX * NY * Q, 0.0);
    f_i.assign(NX * NY * Q, 0.0);
    f_temp_i.assign(NX * NY * Q, 0.0);
    f_eq_i.assign(NX * NY * Q, 0.0);
    //if needed
    f_eq_e_i.assign(NX * NY * Q, 0.0);
    f_eq_i_e.assign(NX * NY * Q, 0.0);

    g_e.assign(NX * NY * Q, 0.0);
    g_i.assign(NX * NY * Q, 0.0);
    g_eq_e.assign(NX * NY * Q, 0.0);
    g_eq_i.assign(NX * NY * Q, 0.0);
    g_temp_e.assign(NX * NY * Q, 0.0);
    g_temp_i.assign(NX * NY * Q, 0.0);

    n_e.assign(NX * NY, 1.0);
    n_i.assign(NX * NY, 1.0);
    ux_e.assign(NX * NY, 0.0);
    uy_e.assign(NX * NY, 0.0);
    ux_i.assign(NX * NY, 0.0);
    uy_i.assign(NX * NY, 0.0);
    ux_e_i.assign(NX * NY, 0.0);
    uy_e_i.assign(NX * NY, 0.0);

    T_e.assign(NX * NY, T_e_SI);
    T_i.assign(NX * NY, T_i_SI);

    phi.assign(NX * NY, 0.0);
    phi_new.assign(NX * NY, 0.0);
    Ex.assign(NX * NY, Ex_latt_init);
    Ey.assign(NX * NY, Ey_latt_init);

    // Net charge density in SI: (ρ_i - ρ_e) * e / dx_SI^3
    rho_q_phys.assign(NX * NY, 0.0); //Should be rho_q_phys[idx] = (n_i[idx] - n_e[idx]) * (e_charge_SI / (dx_SI*dx_SI*dx_SI));
    // In lattice units, store ρ_q_latt = (ρ_i - ρ_e) * 1.0  (just #/cell difference)
    rho_q_latt.assign(NX * NY, 0.0); //Should be rho_q_latt[idx] = (n_i[idx] - n_e[idx]) 

    // 9) Initialize fields:  set f = f_eq(rho=1,u=0), zero φ, E=Ex_latt_init
    Initialize();
    // 10) Print initial values to check
    std::cout
        << "Lx_SI = "<< Lx_SI << " m, Ly_SI = " << Ly_SI << " m\n"
        << "dx_SI = "<< dx_SI << " m, dy_SI = " << dy_SI << " m\n"
        << "dt_SI = "<< dt_SI << " s, dt_dx = "<< dt_dx << " (dimensionless)\n"
        << "T_e_initial = " << T_e_SI << " K, Ti_initial = " << T_i_SI << " K\n"
        << "v_th_e_initial = "<<std::sqrt(kB_SI*T_e_SI/m_e_SI) << " m/s v_th_i_initial = "<< std::sqrt(kB_SI*T_i_SI/m_i_SI)<< " m/s\n"
        << "cs2_e = "<< cs2_e << " cs_e = " << std::sqrt(cs2_e)<< " (latt)\n"
        << "cs2_i = "<< cs2_i << " cs_i = " << std::sqrt(cs2_i)<< " (latt)\n"
        << "q/m_e_SI = "<<-e_charge_SI/m_e_SI<< " q/m_i_SI = "<<Z_ion*e_charge_SI/m_i_SI<<"\n"
        << "q/m_e_latt = "<<qom_e_latt<< " q/m_i_latt = "<<qom_i_latt<<"\n"
        << "Ex_ext_SI = "<<Ex_SI<< " V/m, Ey_ext_latt = "<<Ey_SI<< " V/m\n"
        << "Ex_ext_latt = "<<Ex_latt_init<< " Ey_ext_latt = "<<Ey_latt_init<< "\n"
        <<std::endl;
}

//──────────────────────────────────────────────────────────────────────────────
//  Initialize all fields at t=0:
//    • densities = 1.0 (per cell), velocities = 0
//    • f_e = f_e^eq(ρ=1,u=0), f_i = f_i^eq(ρ=1,u=0)
//    • g_e = g_e^eq = w * T_e, g_i = g_i^eq = w * T_i
//    • phi = 0 everywhere
//    • Ex, Ey = Ex_latt_init, Ey_latt_init
//    • ρ_q_phys = (ρ_i - ρ_e) * (e_charge_SI / dx_SI^3)  → initial net charge  (here 0)
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::Initialize() {
    // Initialize f=f_eq=weight at (ρ=1, u=0)
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < NX * NY * Q; ++idx) {
        const size_t i = idx % Q; // Use size_t for consistency
        f_e[idx] = f_eq_e[idx] = w[i];
        f_i[idx] = f_eq_i[idx] = w[i];

        f_eq_e_i[idx] = f_eq_i_e[idx] = w[i]; // Initialize interaction equilibrium functions
        ////Do all in the same line
        g_e[idx] = g_eq_e[idx] = w[i] * T_e_SI;
        g_i[idx] = g_eq_i[idx] = w[i] * T_i_SI;

    }
}
//──────────────────────────────────────────────────────────────────────────────
//  Compute D2Q9 equilibrium for a given (ρ, ux, uy, T) and sound‐speed cs2.
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::computeEquilibrium() {//It's the same for all the species, maybe can be used as a function
    // Compute the equilibrium distribution function f_eq
    #pragma omp parallel for collapse(2) schedule(static)
        for (size_t x = 0; x < NX; ++x) {
            for (size_t y = 0; y < NY; ++y) {
                const size_t idx = INDEX(x, y); // Get 1D index for 2D point (x, y)
                const double u2_e = ux_e[idx] * ux_e[idx] + uy_e[idx] * uy_e[idx]; // Square of the speed magnitude
                const double u2_i = ux_i[idx] * ux_i[idx] + uy_i[idx] * uy_i[idx]; 
                const double u2_e_i= ux_e_i[idx]*ux_e_i[idx]+uy_e_i[idx]*uy_e_i[idx];
                
                for (size_t i = 0; i < Q; ++i) {
                    const size_t idx_3=INDEX(x, y, i);
                    const double cu_e = cx[i]*ux_e[idx] +cy[i]*uy_e[idx]; // Dot product (c_i · u)
                    const double cu_i = cx[i]*ux_i[idx] +cy[i]*uy_i[idx];
                    const double cu_e_i = cx[i]*ux_e_i[idx] +cy[i]*uy_e_i[idx];
                    // Compute f_eq from discretization of Maxwell Boltzmann distribution function
                    f_eq_e[idx_3]= w[i]*n_e[idx]*(
                        1.0 +
                        (cu_e / cs2_e) +
                        (cu_e * cu_e) / (2.0 * cs2_e * cs2_e) -
                        u2_e / (2.0 * cs2_e)
                    );
                    f_eq_i[idx_3]= w[i]*n_i[idx]*(
                        1.0 +
                        (cu_i / cs2_i) +
                        (cu_i * cu_i) / (2.0 * cs2_i * cs2_i) -
                        u2_i / (2.0 * cs2_i)
                    );
                    f_eq_e_i[idx_3]= w[i]*n_e[idx]*(/// Need to revise to correct cs2_e to cs2_e_i
                        1.0 +
                        (cu_e_i / cs2_e) +
                        (cu_e_i * cu_e_i) / (2.0 * cs2_e * cs2_e) -
                        u2_e_i / (2.0 * cs2_e)
                    );
                    f_eq_i_e[idx_3]= w[i]*n_i[idx]*(/// Need to revise to correct cs2_e to cs2_e_i
                        1.0 +
                        (cu_e_i / cs2_e) +
                        (cu_e_i * cu_e_i) / (2.0 * cs2_e * cs2_e) -
                        u2_e_i / (2.0 * cs2_e)
                    );

                    g_eq_e[idx_3]=w[i]*T_e[idx]*(
                        1.0 +
                        (cu_e / cs2_e) +
                        (cu_e * cu_e ) / (2.0 * cs2_e *cs2_e) -
                        u2_e / (2.0 * cs2_e)
                    );
                    g_eq_i[idx_3]=w[i]*T_i[idx]*(
                        1.0 +
                        (cu_i / cs2_i) +
                        (cu_i * cu_i ) / (2.0 * cs2_i *cs2_i) -
                        u2_i / (2.0 * cs2_i)
                    );
                }
            }
        }
}
//──────────────────────────────────────────────────────────────────────────────
//  Update macroscopic variables for both species:
//    ρ = Σ_i f_i,
//    ρ u = Σ_i f_i c_i + (1/2)*F
//  where F = qom_latt * (Ex_cell, Ey_cell)
//    T = Σ_i g_i
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::UpdateMacro() {
    //Find a way to evaluete the temperature I don't know how to do it
    //maybe from another distribution function g that has to be imposed and solved
    double n_loc_e = 0.0;
    double ux_loc_e = 0.0;
    double uy_loc_e = 0.0;
    double T_loc_e = 0.0;

    double n_loc_i = 0.0;
    double ux_loc_i = 0.0;
    double uy_loc_i = 0.0;
    double T_loc_i = 0.0;

    //const double sigma_el_ion=M_PI*(r_el+r_ion)*(r_el+r_ion); //Cross section for electron-ion interaction
    //const double mean_vel_el = std::sqrt(8*kb/(m_el*M_PI)); //Mean velocity of electrons without Temperature dependence
    //const double mean_vel_ion = std::sqrt(8*kb/(A_ion*m_p*M_PI)); //Mean velocity of ions without Temperature dependence
    //const double mass_el_ion = m_el * A_ion * m_p /(m_el + A_ion * m_p); //Reduced mass for electron-ion interaction
    #pragma omp parallel for collapse(2) private(n_loc_e, ux_loc_e, uy_loc_e, T_loc_e, n_loc_i, ux_loc_i, uy_loc_i, T_loc_i)
        for (size_t x=0; x<NX; ++x){
            for (size_t y = 0; y < NY; ++y) {
                const size_t idx = INDEX(x, y);
                n_loc_e = 0.0;
                ux_loc_e = 0.0;
                uy_loc_e = 0.0;
                T_loc_e = 0.0;

                n_loc_i = 0.0;
                ux_loc_i = 0.0;
                uy_loc_i = 0.0;
                T_loc_i = 0.0;


                for (size_t i = 0; i < Q; ++i) {
                    const size_t idx_3 = INDEX(x, y, i);
                    const double fi_e=f_e[idx_3];
                    n_loc_e += fi_e;
                    ux_loc_e += fi_e * cx[i];
                    uy_loc_e += fi_e * cy[i];
                    T_loc_e += g_e[idx_3];

                    const double fi_i=f_i[idx_3];
                    n_loc_i += fi_i;
                    ux_loc_i += fi_i * cx[i];
                    uy_loc_i += fi_i * cy[i];
                    T_loc_i += g_i[idx_3];
                    
                }
                if (n_loc_e<1e-10){//in order to avoid instabilities (can be removed if the code is stable)
                    n_e[idx] = 0.0;
                    ux_e[idx] = 0.0;
                    uy_e[idx] = 0.0;
                    T_e[idx] = 0.0;
                }else if (n_loc_i<1e-10){
                    n_i[idx] = 0.0;
                    ux_i[idx] = 0.0;
                    uy_i[idx] = 0.0;
                    T_i[idx] = 0.0;
                }
                else {
                    n_e[idx] = n_loc_e;
                    n_i[idx] = n_loc_i;

                    //Add the force term to velocity
                    ux_loc_e += 0.5 * (qom_e_latt * Ex[idx]);
                    uy_loc_e += 0.5 * (qom_e_latt * Ey[idx]);
                    ux_loc_i += 0.5 * (qom_i_latt * Ex[idx]);
                    uy_loc_i += 0.5 * (qom_i_latt * Ey[idx]);

                    //assign velocity
                    ux_e[idx] = ux_loc_e / n_loc_e;
                    uy_e[idx] = uy_loc_e / n_loc_e;
                    ux_i[idx] = ux_loc_i / n_loc_i;
                    uy_i[idx] = uy_loc_i / n_loc_i;

                    // Assign Temperature
                    T_e[idx] = T_loc_e;
                    T_i[idx] = T_loc_i; 

                    /////Other parameters not yet implemented
                    //Relaxation times
                    //tau_el[idx] = 1.0/(M_PI*(2*r_el)*(2*r_el) * n_local_el * mean_vel_el * std::sqrt(T_el[idx])); //Relaxation time for electron
                    //tau_ion[idx] = 1.0/(M_PI*(2*r_ion)*(2*r_ion) * n_local_ion * mean_vel_ion * std::sqrt(T_ion[idx])); //Relaxation time for ion
    
                    //Interaction velocities
                    //ux_el_ion[idx] = (m_el * n_local_el * ux_el[idx] + A_ion * m_p * n_local_ion * ux_ion[idx]) / (m_el *n_local_el + A_ion * m_p * n_local_ion);
                    //uy_el_ion[idx] = (m_el * n_local_el * uy_el[idx] + A_ion * m_p * n_local_ion * uy_ion[idx]) / (m_el *n_local_el + A_ion * m_p * n_local_ion);

                    //Interaction relaxation times
                    //tau_el_ion[idx] = 1.0/(sigma_el_ion * n_local_ion * mean_vel_el * std::sqrt(T_el[idx]));
                    //tau_ion_el[idx] = 1.0/(sigma_el_ion * n_local_el * mean_vel_ion * std::sqrt(T_ion[idx]));
                }
                // Update physical charge density [C/m³]:
                rho_q_phys[idx] =  (n_i[idx] - n_e[idx]) * (e_charge_SI / (dx_SI*dx_SI*dx_SI));
                // Lattice‐unit charge density (#/cell difference):
                rho_q_latt[idx] = (n_i[idx] - n_e[idx]);
            }
        }
}
//──────────────────────────────────────────────────────────────────────────────
//  Poisson dispatcher:
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::SolvePoisson() {
    if (poisson_type == PoissonType::GAUSS_SEIDEL)  SolvePoisson_GS();
    else if (poisson_type == PoissonType::SOR)      SolvePoisson_SOR();
    else if (poisson_type == PoissonType::FFT)      SolvePoisson_fft();
    // else PoissonType::NONE → leave Ex, Ey = Ex_latt_init, Ey_latt_init.
}
//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: Gauss–Seidel with Dirichlet φ=0 on boundary.
//    We solve  ∇² φ = − ρ_q_phys / ε₀,  in lattice units.
//    Our RHS in “lattice‐Land” is:  RHS_latt = − (ρ_q_phys/ε₀) * dx_SI.
//    Then φ_new[i,j] = ¼ [ φ[i+1,j] + φ[i−1,j] + φ[i,j+1] + φ[i,j−1] − RHS_latt[i,j] ].
//
//  After convergence, we reconstruct E with centered differences:
//    E_x = −(φ[i+1,j] − φ[i−1,j]) / 2, etc.
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::SolvePoisson_GS() {
    std::vector<double> RHS(NX*NY, 0.0);

    // Build RHS_latt = −(ρ_q_phys / ε₀) * dx_SI
    for(size_t idx=0; idx<NX*NY; ++idx) {
        RHS[idx] = - (rho_q_phys[idx] / epsilon0_SI) * dx_SI * dx_SI;
    }
    // Initialize φ = 0 everywhere
    for(size_t idx=0; idx<NX*NY; ++idx) {
        phi[idx] = 0.0;
    }

    // GS iterations
    const size_t maxIter = 2000;
    const double tol = 1e-8;
    for(size_t iter=0; iter<maxIter; ++iter) {
        double maxErr = 0.0;
        for(size_t j=1; j<NY-1; ++j) {
            for(size_t i=1; i<NX-1; ++i) {
                size_t idx = INDEX(i,j);
                double old = phi[idx];
                double nbSum = phi[INDEX(i+1,j)]
                             + phi[INDEX(i-1,j)]
                             + phi[INDEX(i,j+1)]
                             + phi[INDEX(i,j-1)];
                phi[idx] = 0.25 * (nbSum - RHS[idx]);
                double err = std::abs(phi[idx] - old);
                if (err > maxErr) maxErr = err;
            }
        }
        if (maxErr < tol) break;
    }

    // Reconstruct E_x, E_y inside:
    for(size_t j=1; j<NY-1; ++j) {
        for(size_t i=1; i<NX-1; ++i) {
            size_t idx = INDEX(i,j);
            Ex[idx] = - (phi[INDEX(i+1,j)] - phi[INDEX(i-1,j)]) / 2.0 + Ex_latt_init;
            Ey[idx] = - (phi[INDEX(i,j+1)] - phi[INDEX(i,j-1)]) / 2.0 + Ey_latt_init;
        }
    }
    // Boundaries: copy neighbors (zero‐Neumann on φ edges)
    for(size_t i=0; i<NX; ++i) {
        Ex[INDEX(i,0)]     = Ex[INDEX(i,1)];
        Ey[INDEX(i,0)]     = Ey[INDEX(i,1)];
        Ex[INDEX(i,NY-1)]  = Ex[INDEX(i,NY-2)];
        Ey[INDEX(i,NY-1)]  = Ey[INDEX(i,NY-2)];
    }
    for(size_t j=0; j<NY; ++j) {
        Ex[INDEX(0,j)]     = Ex[INDEX(1,j)];
        Ey[INDEX(0,j)]     = Ey[INDEX(1,j)];
        Ex[INDEX(NX-1,j)]  = Ex[INDEX(NX-2,j)];
        Ey[INDEX(NX-1,j)]  = Ey[INDEX(NX-2,j)];
    }
}

//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: SOR (over‐relaxed Gauss–Seidel).  Identical 5‐point
//  stencil as GS, but φ_new = (1−ω) φ_old + ω φ_GS.  Stop on tol.
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::SolvePoisson_SOR() {
    std::vector<double> RHS(NX*NY, 0.0);

    for(size_t idx=0; idx<NX*NY; ++idx) {
        RHS[idx] = - (rho_q_phys[idx] / epsilon0_SI) * dx_SI *dx_SI;
    }
    for(size_t idx=0; idx<NX*NY; ++idx) {
        phi[idx] = 0.0;
    }

    const size_t maxIter = 2000;
    const double tol = 1e-8;
    for(size_t iter=0; iter<maxIter; ++iter) {
        double maxErr = 0.0;
        for(size_t j=1; j<NY-1; ++j) {
            for(size_t i=1; i<NX-1; ++i) {
                size_t idx = INDEX(i,j);
                double old = phi[idx];
                double nbSum = phi[INDEX(i+1,j)]
                             + phi[INDEX(i-1,j)]
                             + phi[INDEX(i,j+1)]
                             + phi[INDEX(i,j-1)];
                double phi_GS = 0.25 * (nbSum - RHS[idx]);
                phi[idx] = (1.0 - omega_sor) * old + omega_sor * phi_GS;
                double err = std::abs(phi[idx] - old);
                if (err > maxErr) maxErr = err;
            }
        }
        if (maxErr < tol) break;
    }

    for(size_t j=1; j<NY-1; ++j) {
        for(size_t i=1; i<NX-1; ++i) {
            size_t idx = INDEX(i,j);
            Ex[idx] = - (phi[INDEX(i+1,j)] - phi[INDEX(i-1,j)]) / 2.0 + Ex_latt_init;
            Ey[idx] = - (phi[INDEX(i,j+1)] - phi[INDEX(i,j-1)]) / 2.0 + Ey_latt_init;
        }
    }
    for(size_t i=0; i<NX; ++i) {
        Ex[INDEX(i,0)]     = Ex[INDEX(i,1)];
        Ey[INDEX(i,0)]     = Ey[INDEX(i,1)];
        Ex[INDEX(i,NY-1)]  = Ex[INDEX(i,NY-2)];
        Ey[INDEX(i,NY-1)]  = Ey[INDEX(i,NY-2)];
    }
    for(size_t j=0; j<NY; ++j) {
        Ex[INDEX(0,j)]     = Ex[INDEX(1,j)];
        Ey[INDEX(0,j)]     = Ey[INDEX(1,j)];
        Ex[INDEX(NX-1,j)]  = Ex[INDEX(NX-2,j)];
        Ey[INDEX(NX-1,j)]  = Ey[INDEX(NX-2,j)];
    }
}
void LBmethod::SolvePoisson_fft() {
    //Periodic conditions ONLY

    const double inveps = 1.0 / epsilon0_SI;

    // Allocate FFTW arrays
    fftw_complex *rho_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NX * NY);
    fftw_complex *phi_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NX * NY);

    fftw_plan forward_plan = fftw_plan_dft_r2c_2d(NX, NY, phi.data(), rho_hat, FFTW_ESTIMATE);
    fftw_plan backward_plan = fftw_plan_dft_c2r_2d(NX, NY, phi_hat, phi.data(), FFTW_ESTIMATE);


    // Copy rho_c into phi temporarily to use r2c FFT (input must be real)
    std::copy(rho_q_phys.begin(), rho_q_phys.end(), phi.begin());

    // FFT forward: rho_hat = FFT[rho_c]
    fftw_execute_dft_r2c(forward_plan, phi.data(), rho_hat);

    // Solve in Fourier space
    for (size_t i = 0; i < NX; ++i) {
        int kx = (i <= NX/2) ? i : (int)i - NX;
        double kx2 = 4.0 * std::sin(M_PI * kx / NX) * std::sin(M_PI * kx / NX);

        for (size_t j = 0; j < NY; ++j) {
            int ky = (j <= NY/2) ? j : (int)j - NY;
            double ky2 = 4.0 * std::sin(M_PI * ky / NY) * std::sin(M_PI * ky / NY);

            double denom = (kx2 + ky2);
            const size_t idx = i * NY + j;

            if (denom != 0.0) {
                const double scale = -1.0 / (inveps * denom);
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
        total_charge += rho_q_phys[idx];
    }
    std::cout << "Total net charge in domain: " << total_charge << " C" << std::endl;
    //Then it should compute the total electric field in every region
}

//──────────────────────────────────────────────────────────────────────────────
//  Collision step (BGK + Guo forcing) for both species:
//    f_e_post = f_e - (1/τ_e)(f_e - f_e^eq) + F_e
//    f_i_post = f_i - (1/τ_i)(f_i - f_i^eq) + F_i
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::Collisions() {
    #pragma omp parallel for collapse(2)
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
            const size_t idx = INDEX(x, y);
            
            const double Ex_loc = Ex[idx];
            const double Ey_loc = Ey[idx];

            for (size_t i = 0; i < Q; ++i) {
                const size_t idx_3 = INDEX(x, y, i);
                
                const double F_e = w[i] * qom_e_latt / cs2_e * (1.0-1.0/(2*tau_e_latt)) * (
                    (cx[i]*Ex_loc+cy[i]*Ey_loc)+
                    (cx[i]*ux_e[idx]+cy[i]*uy_e[idx])/cs2_e-
                    (ux_e[idx]*Ex_loc+uy_e[idx]*Ey_loc)
                );//ADJUST    F=w*(1-1/(2*tau))*(c*F/cs2+(c*u)*(c*F)/cs2^2-u*F/cs2)
                const double F_i = w[i] * qom_i_latt / cs2_i * (1.0-1.0/(2*tau_i_latt)) * (
                    (cx[i]*Ex_loc+cy[i]*Ey_loc)+
                    (cx[i]*ux_i[idx]+cy[i]*uy_i[idx])/cs2_i-
                    (ux_i[idx]*Ex_loc+uy_i[idx]*Ey_loc)
                );//maybe simplify a bit the writing
               
                // Compute complete collisions terms
                const double C_e = -(f_e[idx_3]-f_eq_e[idx_3]) / tau_e_latt;
                const double C_i = -(f_i[idx_3]-f_eq_i[idx_3]) / tau_i_latt;

                // Update distribution functions with Guo forcing term
                f_e[idx_3]  += C_e + F_e;
                f_i[idx_3]  += C_i + F_i;
            }
        }
    }
}
//──────────────────────────────────────────────────────────────────────────────
//  Thermal Collision step for both species:
//    g_e_post = g_e - (1/τ_Te)(g_e - g_e^eq) + Source
//    g_i_post = g_i - (1/τ_Ti)(g_i - g_i^eq) + Source
//  Now no Source is added
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::ThermalCollisions() {
    #pragma omp parallel for collapse(3)
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
            for (size_t i = 0; i < Q; ++i) {
                const size_t idx_3 = INDEX(x, y, i);
                // Compute complete collisions terms
                const double C_e = -(g_e[idx_3]-g_eq_e[idx_3]) / tau_Te_latt;
                const double C_i = -(g_i[idx_3]-g_eq_i[idx_3]) / tau_Ti_latt;

                // Update distribution functions with Guo forcing term
                g_e[idx_3]  += C_e;
                g_i[idx_3]  += C_i;
            }
        }
    }
}
//──────────────────────────────────────────────────────────────────────────────
//  Streaming dispatcher:
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::Streaming() {
    if (bc_type == BCType::PERIODIC)       Streaming_Periodic();
    else /* BCType::BOUNCE_BACK */         Streaming_BounceBack();
}
//──────────────────────────────────────────────────────────────────────────────
//  Streaming + PERIODIC boundaries:
//    f_{i}(x + c_i, y + c_i, t+1) = f_{i}(x,y,t)
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::Streaming_Periodic() {
    // f(x,y,t+1) = f(x-cx, y-cy, t) con condizioni periodiche
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
            for (size_t i = 0; i < Q; ++i) {
                // Streaming coordinates (periodic wrapping)
                size_t x_str = (x + NX + cx[i]) % NX;
                size_t y_str = (y + NY + cy[i]) % NY;

                f_temp_i[INDEX(x_str, y_str, i)] = f_i[INDEX(x, y, i)];
                f_temp_e[INDEX(x_str, y_str, i)]  = f_e[INDEX(x, y, i)];
            }
        }
    }
    f_e.swap(f_temp_e);
    f_i.swap(f_temp_i);
}
//──────────────────────────────────────────────────────────────────────────────
//  Streaming + BOUNCE‐BACK walls at x=0, x=NX−1, y=0, y=NY−1.
//    If (x + c_i,x) or (y + c_i,y) is out‐of‐bounds, reflect:
//       f_{i*}(x,y) += f_{i}(x,y),
//    where i* = opp[i].
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::Streaming_BounceBack() {
    //f(x,y,t+1)=f(x-cx,y-cy,t)
    // Parallelize the streaming step (over x and y)
    #pragma omp parallel  ////Needed for???
    {
        #pragma omp for collapse(3) schedule(static)
        for (size_t x = 0; x < NX; ++x) {
            for (size_t y = 0; y < NY; ++y) {
                for (size_t i = 0; i < Q; ++i) {
                    //Define streaming coordinate
                    const int x_str = x + cx[i];
                    const int y_str = y + cy[i];
                    if (x_str >= 0 && x_str <static_cast<int>(NX) && y_str >= 0 && y_str <static_cast<int>(NY)) {
                        //We are inside the lattice so simple streaming
                        f_temp_e[INDEX(x_str, y_str, i)] = f_e[INDEX(x, y, i)];
                        f_temp_i[INDEX(x_str, y_str, i)] = f_i[INDEX(x, y, i)];
                    }
                    else{
                        //We are outside so bounceback
                        f_temp_e[INDEX(x,y,opp[i])]=f_e[INDEX(x,y,i)];
                        f_temp_i[INDEX(x,y,opp[i])]=f_i[INDEX(x,y,i)];
                    }
                }
            }
        }
    }
    f_e.swap(f_temp_e);
    f_i.swap(f_temp_i);
    //f_temp is f at t=t+1 so now we use the new function f_temp in f
}
//──────────────────────────────────────────────────────────────────────────────
//  Streaming + BOUNCE‐BACK walls at x=0, x=NX−1, y=0, y=NY−1.
//    If (x + c_i,x) or (y + c_i,y) is out‐of‐bounds, reflect:
//       g_{i*}(x,y) += g_{i}(x,y),
//    where i* = opp[i].
//  That condition correspond to a Neumann BC so to a zero normal flux at wall
//──────────────────────────────────────────────────────────────────────────────
//Other possibilities: 
//Periodic: when temperature at extreme are fixed and equal for opposite site
//Dirichlet: when temperture at border are fixed
//Cauchy: -kdT/dn=q when the bounce back lose energy 
void LBmethod::ThermalStreaming_BounceBack() {
    #pragma omp for collapse(3) schedule(static)
        for (size_t x = 0; x < NX; ++x) {
            for (size_t y = 0; y < NY; ++y) {
                for (size_t i = 0; i < Q; ++i) {
                    //Define streaming coordinate
                    const int x_str = x + cx[i];
                    const int y_str = y + cy[i];
                    if (x_str >= 0 && x_str <static_cast<int>(NX) && y_str >= 0 && y_str <static_cast<int>(NY)) {
                        //We are inside the lattice so simple streaming
                        g_temp_e[INDEX(x_str, y_str, i)] = g_e[INDEX(x, y, i)];
                        g_temp_i[INDEX(x_str, y_str, i)] = g_i[INDEX(x, y, i)];
                    }
                    else{
                        //We are outside so bounceback
                        g_temp_e[INDEX(x,y,opp[i])]=g_e[INDEX(x,y,i)];
                        g_temp_i[INDEX(x,y,opp[i])]=g_i[INDEX(x,y,i)];
                    }
                }
            }
        }
    g_e.swap(g_temp_e);
    g_i.swap(g_temp_i);
}
void LBmethod::Run_simulation() {
    
    // Set threads for this simulation
    omp_set_num_threads(n_cores);


    // Pre‐compute the frame sizes for each video:
    const int border       = 10;
    const int label_height = 30;
    const int tile_w       = NX + 2 * border;                // panel width
    const int tile_h       = NY + 2 * border + label_height; // panel height

    // --- Density‐video (2 panels side by side) ---
    {
        int legend_width = 40, text_area = 60;
        int panel_width  = legend_width + text_area;
        int frame_w = 3 * tile_w + 2 * panel_width + 4 * border;
        int frame_h = tile_h + 2 * border;
        video_writer_density.open(
            "density_video.mp4",
            cv::VideoWriter::fourcc('m','p','4','v'),
            10.0,                        // fps
            cv::Size(frame_w, frame_h),
            true                        // isColor
        );
        if (!video_writer_density.isOpened()) {
            std::cerr << "Cannot open density_video.mp4 for writing\n";
            return;
        }
    }

    // --- Velocity‐video (2 rows × 3 columns = 6 panels) ---
    {
        int legend_width = 40, text_area = 60;
        int panel_width  = legend_width + text_area;  // width of one legend (strip+text)

        // Grid of 3 panels across × 2 panels down:
        int grid_w = 3 * tile_w;
        int grid_h = 2 * tile_h;

        // Because we have TWO legend panels (one at left, one at right),
        // and we keep a 'border' px on every side plus between each block,
        // the total width is:
        //
        //   [ border ] [ left legend width=panel_width ] [ border ]
        //   [    grid width = grid_w           ] [ border ]
        //   [ right legend width=panel_width ] [ border ]
        //
        // That is: 4×border + 2×panel_width + grid_w
        int frame_w = grid_w + 2 * panel_width + 4 * border;

        // Vertically, we keep a 'border' on top and bottom of the entire 2×3 grid:
        //   total height = grid_h + 2×border
        int frame_h = grid_h + 2 * border;

        video_writer_velocity.open(
            "velocity_video.mp4",
            cv::VideoWriter::fourcc('m','p','4','v'),
            10.0,                        // fps
            cv::Size(frame_w, frame_h), // must match exactly what VisualizationVelocity() writes
            true                         // isColor = true
        );
        if (!video_writer_velocity.isOpened()) {
            std::cerr << "Cannot open velocity_video.mp4 for writing\n";
            return;
        }
    }


    // --- Temperature‐video (2 panels side by side) ---
    {
        int legend_width = 40, text_area = 60;
        int panel_width  = legend_width + text_area;
        int frame_w      = 2 * tile_w + panel_width + 3 * border;
        int frame_h      = tile_h + 2 * border;
        video_writer_temperature.open(
            "temperature_video.mp4",
            cv::VideoWriter::fourcc('m','p','4','v'),
            10.0,
            cv::Size(frame_w, frame_h),
            true
        );
        if (!video_writer_temperature.isOpened()) {
            std::cerr << "Cannot open temperature_video.mp4 for writing\n";
            return;
        }
    }

    //──────────────────────────────────────────────────────────────────────────────
    //  Main loop: for t = 0 … NSTEPS−1,
    //    [1] Update macros (ρ, u)
    //    [2] Solve Poisson → update Ex, Ey
    //    [3] Collisions (BGK + forcing)
    //    [4] Streaming (+ BC)
    //    [5] Visualization
    //──────────────────────────────────────────────────────────────────────────────
    for (size_t t=0; t<NSTEPS; ++t){
        if (t%100==0) {
            auto max_ne = std::max_element(n_e.begin(), n_e.end());
            auto min_ne = std::min_element(n_e.begin(), n_e.end());
            auto max_ni = std::max_element(n_i.begin(), n_i.end());
            auto min_ni = std::min_element(n_i.begin(), n_i.end());
            auto max_uxe = std::max_element(ux_e.begin(), ux_e.end());
            auto min_uxe = std::min_element(ux_e.begin(), ux_e.end());
            auto max_uye = std::max_element(uy_e.begin(), uy_e.end());
            auto min_uye = std::min_element(uy_e.begin(), uy_e.end());
            auto max_uxi = std::max_element(ux_i.begin(), ux_i.end());
            auto min_uxi = std::min_element(ux_i.begin(), ux_i.end());
            auto max_uyi = std::max_element(uy_i.begin(), uy_i.end());
            auto min_uyi = std::min_element(uy_i.begin(), uy_i.end());
            auto max_Ex = std::max_element(Ex.begin(), Ex.end());
            auto min_Ex = std::min_element(Ex.begin(), Ex.end());
            auto max_Ey = std::max_element(Ey.begin(), Ey.end());
            auto min_Ey = std::min_element(Ey.begin(), Ey.end());
            auto max_rho = std::max_element(rho_q_latt.begin(), rho_q_latt.end());
            auto min_rho = std::min_element(rho_q_latt.begin(), rho_q_latt.end());
            auto max_Te = std::max_element(T_e.begin(), T_e.end());
            auto min_Te = std::min_element(T_e.begin(), T_e.end());
            auto max_Ti = std::max_element(T_i.begin(), T_i.end());
            auto min_Ti = std::min_element(T_i.begin(), T_i.end());
            std::cout <<"Step:"<<t<<std::endl;
            std::cout <<"max n_e= "<<*max_ne<<", min n_e= "<<*min_ne<<std::endl;
            std::cout <<"max n_i= "<<*max_ni<<", min n_i= "<<*min_ni<<std::endl;
            std::cout <<"max ux_e= "<<*max_uxe<<", min ux_e= "<<*min_uxe<<std::endl;
            std::cout <<"max ux_i= "<<*max_uxi<<", min ux_e= "<<*min_uxi<<std::endl;
            std::cout <<"max uy_e= "<<*max_uye<<", min uy_e= "<<*min_uye<<std::endl;
            std::cout <<"max uy_i= "<<*max_uyi<<", min uy_i= "<<*min_uyi<<std::endl;
            std::cout <<"max Ex= "<<*max_Ex<<", min Ex= "<<*min_Ex<<std::endl;
            std::cout <<"max Ey= "<<*max_Ey<<", min Ey= "<<*min_Ey<<std::endl;
            std::cout <<"max T_e= "<<*max_Te<<", min T_e= "<<*min_Te<<std::endl;
            std::cout <<"max T_i= "<<*max_Ti<<", min T_i= "<<*min_Ti<<std::endl;
            std::cout <<"max rho_q (latt)= "<<*max_rho<<", rho_q (latt)= "<<*min_rho<<std::endl;
            std::cout <<std::endl;
        }
        if(NX< 6){
            std::cout<<"n_e= "<<std::endl;
            for (size_t x = 0; x < NX; ++x) {
                for (size_t y = 0; y < NY; ++y) {
                    std::cout<<n_e[INDEX(x,y)]<< " ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
            std::cout<<"n_i= "<<std::endl;
            for (size_t x = 0; x < NX; ++x) {
                for (size_t y = 0; y < NY; ++y) {
                    std::cout<<n_i[INDEX(x,y)]<< " ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
            std::cout<<"ux_e= "<<std::endl;
            for (size_t x = 0; x < NX; ++x) {
                for (size_t y = 0; y < NY; ++y) {
                    std::cout<<ux_e[INDEX(x,y)]<< " ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
            std::cout<<"uy_e= "<<std::endl;
            for (size_t x = 0; x < NX; ++x) {
                for (size_t y = 0; y < NY; ++y) {
                    std::cout<<uy_e[INDEX(x,y)]<< " ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
            std::cout<<"ux_i= "<<std::endl;
            for (size_t x = 0; x < NX; ++x) {
                for (size_t y = 0; y < NY; ++y) {
                    std::cout<<ux_i[INDEX(x,y)]<< " ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
            std::cout<<"uy_i= "<<std::endl;
            for (size_t x = 0; x < NX; ++x) {
                for (size_t y = 0; y < NY; ++y) {
                    std::cout<<uy_i[INDEX(x,y)]<< " ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
            std::cout<<"rho_phys= "<<std::endl;
            for (size_t x = 0; x < NX; ++x) {
                for (size_t y = 0; y < NY; ++y) {
                    std::cout<<rho_q_phys[INDEX(x,y)]<< " ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
            std::cout<<"rho latt= "<<std::endl;
            for (size_t x = 0; x < NX; ++x) {
                for (size_t y = 0; y < NY; ++y) {
                    std::cout<<rho_q_latt[INDEX(x,y)]<< " ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
                std::cout<<"Ex= "<<std::endl;
            for (size_t x = 0; x < NX; ++x) {
                for (size_t y = 0; y < NY; ++y) {
                    std::cout<<Ex[INDEX(x,y)]<< " ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
            std::cout<<"Ey= "<<std::endl;
            for (size_t x = 0; x < NX; ++x) {
                for (size_t y = 0; y < NY; ++y) {
                    std::cout<<Ey[INDEX(x,y)]<< " ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;

            std::cout<<"Te= "<<std::endl;
            for (size_t x = 0; x < NX; ++x) {
                for (size_t y = 0; y < NY; ++y) {
                    std::cout<<T_e[INDEX(x,y)]<< " ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
            std::cout<<"Ti= "<<std::endl;
            for (size_t x = 0; x < NX; ++x) {
                for (size_t y = 0; y < NY; ++y) {
                    std::cout<<T_i[INDEX(x,y)]<< " ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }  
        UpdateMacro(); // rho=sum(f), ux=sum(f*c_x)/rho, uy=sum(f*c_y)/rho
        computeEquilibrium();
        SolvePoisson();
        Collisions(); // f(x,y,t+1)=f(x-cx,y-cy,t) + tau * (f_eq - f) + dt*F
        ThermalCollisions();
        Streaming(); // f(x,y,t+1)=f(x-cx,y-cy,t)
        ThermalStreaming_BounceBack();

        VisualizationDensity(t);
        VisualizationVelocity(t);
        VisualizationTemperature(t);
    }

    video_writer_density.release();
    video_writer_velocity.release();
    video_writer_temperature.release();
    std::cout << "Video saved, simulation ended " << std::endl;
}
//──────────────────────────────────────────────────────────────────────────────
//  Visualization stub.  Use OpenCV to save density images at time t, etc.
//──────────────────────────────────────────────────────────────────────────────

// --------------------------------------------------------------
// Visualization for density
// --------------------------------------------------------------

void LBmethod::VisualizationDensity(size_t t) {
    // ───────────────────────────────────────────────────────────────
    // 1) Layout constants (must match main()'s video‐writer dimensions)
    // ───────────────────────────────────────────────────────────────
    constexpr int border       = 10;  // white margin on all sides
    constexpr int label_height = 30;  // space below each heatmap for its text label

    // ───────────────────────────────────────────────────────────────
    // 3) Allocate (on first call) floating‐point mats for n_e, n_i, ρ_q
    // ───────────────────────────────────────────────────────────────
    static cv::Mat density_mat_i, density_mat_e, rho_q_mat;
    static cv::Mat density_heatmap_i, density_heatmap_e, rho_q_heatmap;
    static cv::Mat wrapped_i, wrapped_e, wrapped_rho_q;
    static cv::Mat grid;
    static cv::Mat output_frame;

    if (t == 0) {
        density_mat_i = cv::Mat(NY, NX, CV_32F);
        density_mat_e = cv::Mat(NY, NX, CV_32F);
        rho_q_mat     = cv::Mat(NY, NX, CV_32F);
    }

    // ───────────────────────────────────────────────────────────────
    // 4) Copy data into mats (parallelized)
    // ───────────────────────────────────────────────────────────────
    #pragma omp parallel for collapse(2)
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
            size_t idx = INDEX(x, y);
            density_mat_i.at<float>(y, x) = static_cast<float>(n_i[idx]);
            density_mat_e.at<float>(y, x) = static_cast<float>(n_e[idx]);
            rho_q_mat.at<float>(y, x)     = static_cast<float>(rho_q_latt[idx]);
        }
    }
    // ───────────────────────────────────────────────────────────────
    // 5) Define constant values
    // ───────────────────────────────────────────────────────────────
    constexpr double density_min = 0.99999999, density_max = +1.0;
    double d_buff=1e9;
    constexpr double charge_min = -1.0*1e-10, charge_max = +1.0*1e-10;
    // ───────────────────────────────────────────────────────────────
    // 6) Normalize n_e, n_i into [0…255] using fixed global range
    // ───────────────────────────────────────────────────────────────
    cv::Mat norm_i, norm_e;
    density_mat_i.convertTo(
        norm_i, CV_8U,
        255.0 / (density_max - density_min),
        -255.0 * density_min / (density_max - density_min)
    );
    density_mat_e.convertTo(
        norm_e, CV_8U,
        255.0 / (density_max - density_min),
        -255.0 * density_min / (density_max - density_min)
    );

    // ───────────────────────────────────────────────────────────────
    // 7) Normalize ρ_q into [0…255] using fixed charge range
    // ───────────────────────────────────────────────────────────────
    cv::Mat norm_rho_q;
    rho_q_mat.convertTo(
        norm_rho_q, CV_8U,
        255.0 / (charge_max - charge_min),
        -255.0 * charge_min / (charge_max - charge_min)
    );

    // ───────────────────────────────────────────────────────────────
    // 8) Apply JET colormap & flip vertically
    // ───────────────────────────────────────────────────────────────
    cv::applyColorMap(norm_i, density_heatmap_i, cv::COLORMAP_JET);
    cv::applyColorMap(norm_e, density_heatmap_e, cv::COLORMAP_JET);
    cv::applyColorMap(norm_rho_q, rho_q_heatmap, cv::COLORMAP_JET);

    cv::flip(density_heatmap_i, density_heatmap_i, 0);
    cv::flip(density_heatmap_e, density_heatmap_e, 0);
    cv::flip(rho_q_heatmap,     rho_q_heatmap,     0);

    // ───────────────────────────────────────────────────────────────
    // 9) Helper: wrap a heatmap with a white border + bottom label
    // ───────────────────────────────────────────────────────────────
    auto wrap_with_label = [&](const cv::Mat& src, const std::string& label_text) {
        cv::Mat bordered;
        cv::copyMakeBorder(
            src,
            bordered,
            border,
            border + label_height,
            border,
            border,
            cv::BORDER_CONSTANT,
            cv::Scalar(255, 255, 255)
        );
        cv::putText(
            bordered,
            label_text,
            cv::Point(border + 5, bordered.rows - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 0, 0),
            1
        );
        return bordered;
    };

    wrapped_e     = wrap_with_label(density_heatmap_e, "n Electrons");
    wrapped_rho_q = wrap_with_label(rho_q_heatmap,     "rho_q");
    wrapped_i     = wrap_with_label(density_heatmap_i, "n Ions");

    // ───────────────────────────────────────────────────────────────
    // 10) Build a single‐row grid of [n_e | ρ_q | n_i]
    // ───────────────────────────────────────────────────────────────
    cv::hconcat(
        std::vector<cv::Mat>{ wrapped_e, wrapped_rho_q, wrapped_i },
        grid
    );

    // ───────────────────────────────────────────────────────────────
    // 11) Prepare right‐side legend for density (n_e & n_i)
    // ───────────────────────────────────────────────────────────────
    int legend_width  = 40;
    int legend_height = grid.rows - 2 * border;
    cv::Mat legend_gray(legend_height, 1, CV_8U);
    for (int i = 0; i < legend_height; ++i) {
        legend_gray.at<uchar>(i, 0) =
            static_cast<uchar>(255 - (i * 255 / (legend_height - 1)));
    }
    cv::Mat legend_color;
    cv::applyColorMap(legend_gray, legend_color, cv::COLORMAP_JET);
    cv::resize(legend_color, legend_color,
               cv::Size(legend_width, legend_height),
               0, 0, cv::INTER_NEAREST);

    int text_margin = 5;
    int density_panel_width = legend_width + 60; // 60 px for numeric labels
    cv::Mat density_legend_panel(
        legend_height + 2 * border,
        density_panel_width,
        CV_8UC3,
        cv::Scalar(255, 255, 255)
    );
    cv::Rect density_legend_roi(border, border, legend_width, legend_height);
    legend_color.copyTo(density_legend_panel(density_legend_roi));

    // Numeric labels for density legend
    cv::Scalar text_color(0, 0, 0);
    int font      = cv::FONT_HERSHEY_SIMPLEX;
    double fscale = 0.5;
    int thickness = 1;
    int baseline;

    std::ostringstream oss_den_max, oss_den_mid, oss_den_min;
    double d_min=density_min * d_buff;
    double d_mid=((density_min+density_max)/2.0) * d_buff;
    oss_den_max << std::fixed << std::setprecision(1) << density_max;
    oss_den_min << std::fixed << std::setprecision(1) << d_min;
    oss_den_mid << std::fixed << std::setprecision(1) << d_mid;

    cv::Size ts_den_max = cv::getTextSize(oss_den_max.str(), font, fscale, thickness, &baseline);
    cv::Size ts_den_mid = cv::getTextSize(oss_den_mid.str(), font, fscale, thickness, &baseline);
    cv::Size ts_den_min = cv::getTextSize(oss_den_min.str(), font, fscale, thickness, &baseline);

    int x_den_text = border + legend_width + text_margin;
    int y_den_max  = border + ts_den_max.height;
    int y_den_mid  = border + (legend_height / 2) + (ts_den_mid.height / 2);
    int y_den_min  = border + legend_height - (ts_den_min.height / 2);

    cv::putText(density_legend_panel, oss_den_max.str(), cv::Point(x_den_text, y_den_max),
                font, fscale, text_color, thickness);
    cv::putText(density_legend_panel, oss_den_mid.str(), cv::Point(x_den_text, y_den_mid),
                font, fscale, text_color, thickness);
    cv::putText(density_legend_panel, oss_den_min.str(), cv::Point(x_den_text, y_den_min),
                font, fscale, text_color, thickness);

    // ───────────────────────────────────────────────────────────────
    // 12) Prepare left‐side legend for charge density (ρ_q) [fixed ±1.5]
    // ───────────────────────────────────────────────────────────────
    cv::Mat charge_gray(legend_height, 1, CV_8U);
    for (int i = 0; i < legend_height; ++i) {
        charge_gray.at<uchar>(i, 0) =
            static_cast<uchar>(255 - (i * 255 / (legend_height - 1)));
    }
    cv::Mat charge_color;
    cv::applyColorMap(charge_gray, charge_color, cv::COLORMAP_JET);
    cv::resize(charge_color, charge_color,
               cv::Size(legend_width, legend_height),
               0, 0, cv::INTER_NEAREST);

    cv::Mat charge_legend_panel(
        legend_height + 2 * border,
        density_panel_width,   // same width as density legend
        CV_8UC3,
        cv::Scalar(255, 255, 255)
    );
    cv::Rect charge_legend_roi(border, border, legend_width, legend_height);
    charge_color.copyTo(charge_legend_panel(charge_legend_roi));

    // One‐line note for charge range: “ρ_q buff”
    std::ostringstream oss_charge_note;
    oss_charge_note << "rho_q*10e10:";
    cv::Size ts_charge_note = cv::getTextSize(oss_charge_note.str(), font, fscale, thickness, &baseline);
    int x_charge_note = border;
    int y_charge_note = (border + ts_charge_note.height) / 2;
    cv::putText(charge_legend_panel, oss_charge_note.str(),
                cv::Point(x_charge_note, y_charge_note),
                font, fscale, text_color, thickness);

    // One‐line note for charge range: “density buff”
    std::ostringstream oss_density_note;
    oss_density_note << "n*10e9:";
    cv::Size ts_density_note = cv::getTextSize(oss_density_note.str(), font, fscale, thickness, &baseline);
    int x_density_note =  border;
    int y_density_note = (border + ts_density_note.height) / 2;
    cv::putText(density_legend_panel, oss_density_note.str(),
                cv::Point(x_density_note, y_density_note),
                font, fscale, text_color, thickness);

    // Numeric labels for charge legend: “+1.50 / 0.00 / -1.50”
    std::ostringstream oss_ch_max, oss_ch_mid, oss_ch_min;
    oss_ch_max << std::fixed << std::setprecision(1) << charge_max;   // “1.50”
    oss_ch_mid << std::fixed << std::setprecision(1) << 0.0;           // “0.00”
    oss_ch_min << std::fixed << std::setprecision(1) << charge_min;   // “-1.50”

    cv::Size ts_ch_max = cv::getTextSize(oss_ch_max.str(), font, fscale, thickness, &baseline);
    cv::Size ts_ch_mid = cv::getTextSize(oss_ch_mid.str(), font, fscale, thickness, &baseline);
    cv::Size ts_ch_min = cv::getTextSize(oss_ch_min.str(), font, fscale, thickness, &baseline);

    int x_ch_text = border + legend_width + text_margin;
    int y_ch_max  = border + ts_ch_max.height;
    int y_ch_mid  = border + (legend_height / 2) + (ts_ch_mid.height / 2);
    int y_ch_min  = border + legend_height - (ts_ch_min.height / 2);

    cv::putText(charge_legend_panel, oss_ch_max.str(), cv::Point(x_ch_text, y_ch_max),
                font, fscale, text_color, thickness);
    cv::putText(charge_legend_panel, oss_ch_mid.str(), cv::Point(x_ch_text, y_ch_mid),
                font, fscale, text_color, thickness);
    cv::putText(charge_legend_panel, oss_ch_min.str(), cv::Point(x_ch_text, y_ch_min),
                font, fscale, text_color, thickness);

    // ───────────────────────────────────────────────────────────────
    // 13) Combine [charge_legend | grid | density_legend] into output_frame
    // ───────────────────────────────────────────────────────────────
    int total_width  = grid.cols + 2 * density_panel_width + 4 * border;
    int total_height = grid.rows + 2 * border;
    output_frame = cv::Mat(total_height, total_width, CV_8UC3, cv::Scalar(255, 255, 255));

    // 13a) Paste charge_legend_panel at (border, border)
    charge_legend_panel.copyTo(output_frame(cv::Rect(
        border,
        border,
        density_panel_width,
        legend_height + 2 * border
    )));

    // 13b) Paste grid at (border + panel_width + border, border)
    int grid_x = border + density_panel_width + border;
    grid.copyTo(output_frame(cv::Rect(
        grid_x,
        border,
        grid.cols,
        grid.rows
    )));

    // 13c) Paste density_legend_panel at (grid_x + grid.cols + border, border)
    int den_leg_x = grid_x + grid.cols + border;
    density_legend_panel.copyTo(output_frame(cv::Rect(
        den_leg_x,
        border,
        density_panel_width,
        legend_height + 2 * border
    )));

    // ───────────────────────────────────────────────────────────────
    // 14) Write this final frame to the density‐video writer
    // ───────────────────────────────────────────────────────────────
    video_writer_density.write(output_frame);
}



// --------------------------------------------------------------
// Visualization for velocity
// --------------------------------------------------------------
void LBmethod::VisualizationVelocity(size_t t) {
    // ───────────────────────────────────────────────────────────────
    // 1) Layout constants (must match what main() uses to open the writer)
    // ───────────────────────────────────────────────────────────────
    constexpr int border       = 10;  // white margin on all edges
    constexpr int label_height = 30;  // space below each heatmap for its text label

    // Each “panel” (one heatmap + border + text strip) is:
    //   panel_w × panel_h pixels, where:
    int panel_w = NX + 2 * border;                   // width of one heatmap + side‐margins
    int panel_h = NY + 2 * border + label_height;     // height of one heatmap + top/bottom margins + label

    // We arrange 3 panels per row, 2 rows total → “grid” is 2×3:
    //   grid_w = 3 * panel_w
    //   grid_h = 2 * panel_h
    int grid_w = 3 * panel_w;
    int grid_h = 2 * panel_h;

    // Legend parameters (same for electrons and ions)
    const int legend_strip_w = 40;  // width of the JET strip itself
    const int text_area_w    = 60;  // extra room on the right for numeric labels / scale note
    const int legend_panel_w = legend_strip_w + text_area_w;  // total width of one legend panel
    // Legend height (strip only) = grid_h – 2⋅border  (we leave “border” px top & bottom)
    int legend_h = grid_h - 4 * border;

    // Final frame dimensions:
    //   horizontally: 
    //     border + (electron‐legend) + border + (grid) + border + (ion‐legend) + border 
    //     = 4*border + 2*legend_panel_w + grid_w
    //   vertically:
    //     border + (grid) + border 
    //     = 2*border + grid_h
    int frame_w = 4 * border + 2 * legend_panel_w + grid_w;
    int frame_h = 2 * border + grid_h;

    // ───────────────────────────────────────────────────────────────
    // 2) “Buff” constants + fixed‐ranges (in raw units):
    //   • We want electrons to span [–1.5e–13, +1.5e–13] in raw units.
    //     Since we multiply (buff) by 1e13, that becomes [–1.5, +1.5] in “buffed” float.
    //   • We want ions to span [–0.5e–18, +0.5e–18] in raw units.
    //     Since we multiply (buff) by 1e18, that becomes [–0.5, +0.5] in “buffed” float.
    //
    // We will normalize each species’ buffed data to [0…255] based on those fixed ends.
    // ───────────────────────────────────────────────────────────────
    const float e_buff     = 1e13f;
    const float i_buff     = 1e18f;
    const double e_min_raw = -7.0e-13;
    const double e_max_raw = +7.0e-13;
    const double i_min_raw = -2.0e-18;
    const double i_max_raw = +2.0e-18;

    // In “buffed” units (after multiplying by e_buff or i_buff),
    //   electron range = [ e_min_raw * 1e13, e_max_raw * 1e13 ] = [ –7.0, +7.0 ]
    //   ion range      = [ i_min_raw * 1e18, i_max_raw * 1e18 ] = [ –2.0, +2.0 ]
    const double e_min = e_min_raw * static_cast<double>(e_buff);  // –7.0
    const double e_max = e_max_raw * static_cast<double>(e_buff);  // +7.0
    const double i_min = i_min_raw * static_cast<double>(i_buff);  // –2.0
    const double i_max = i_max_raw * static_cast<double>(i_buff);  // +2.0

    // ───────────────────────────────────────────────────────────────
    // 3) Allocate mats on first call (floating‐point) for the six fields:
    //   ux_e, uy_e, |u_e|, ux_i, uy_i, |u_i|
    // ───────────────────────────────────────────────────────────────
    static cv::Mat ux_e_mat, uy_e_mat, speed_e_mat;
    static cv::Mat ux_i_mat, uy_i_mat, speed_i_mat;
    if (t == 0) {
        ux_e_mat    = cv::Mat(NY, NX, CV_32F);
        uy_e_mat    = cv::Mat(NY, NX, CV_32F);
        speed_e_mat = cv::Mat(NY, NX, CV_32F);
        ux_i_mat    = cv::Mat(NY, NX, CV_32F);
        uy_i_mat    = cv::Mat(NY, NX, CV_32F);
        speed_i_mat = cv::Mat(NY, NX, CV_32F);
    }

    // ───────────────────────────────────────────────────────────────
    // 4) Fill those mats (parallelized) **buffing** by e_buff/i_buff:
    //   (After this, electron values lie in [–1.5, +1.5], 
    //    ion values lie in [–0.5, +0.5].)
    // ───────────────────────────────────────────────────────────────
    #pragma omp parallel for collapse(2)
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
            size_t idx = INDEX(x,y);

            // Raw fields (assumed in SI units)
            double raw_ux_e = ux_e[idx];
            double raw_uy_e = uy_e[idx];
            double raw_ux_i = ux_i[idx];
            double raw_uy_i = uy_i[idx];

            // “Buff” them:
            float ux_e_buff = static_cast<float>(raw_ux_e * e_buff);
            float uy_e_buff = static_cast<float>(raw_uy_e * e_buff);
            float ux_i_buff = static_cast<float>(raw_ux_i * i_buff);
            float uy_i_buff = static_cast<float>(raw_uy_i * i_buff);

            ux_e_mat.at<float>(y, x)    = ux_e_buff;
            uy_e_mat.at<float>(y, x)    = uy_e_buff;
            ux_i_mat.at<float>(y, x)    = ux_i_buff;
            uy_i_mat.at<float>(y, x)    = uy_i_buff;

            speed_e_mat.at<float>(y, x) = std::sqrt(ux_e_buff*ux_e_buff + uy_e_buff*uy_e_buff);
            speed_i_mat.at<float>(y, x) = std::sqrt(ux_i_buff*ux_i_buff + uy_i_buff*uy_i_buff);
        }
    }

    // ───────────────────────────────────────────────────────────────
    // 5) Normalize each species’ mats to CV_8U, using OUR FIXED range:
    //    • For electrons: map [e_min…e_max] → [0…255]  
    //    • For ions:      map [i_min…i_max] → [0…255]
    // If a buffed‐field value falls outside these ends, it will be clamped
    // to 0 or 255 automatically by convertTo().
    // ───────────────────────────────────────────────────────────────
    float e_scale = 255.0f / static_cast<float>(e_max - e_min);  // 255/(1.5−(−1.5)) = 255/3.0
    float e_shift = -static_cast<float>(e_min * e_scale);       // so that e_min → 0, e_max → 255

    float i_scale = 255.0f / static_cast<float>(i_max - i_min);  // 255/(0.5−(−0.5)) = 255/1.0
    float i_shift = -static_cast<float>(i_min * i_scale);       // so that i_min → 0, i_max → 255

    cv::Mat norm_ue, norm_ve, norm_se;
    cv::Mat norm_ui, norm_vi, norm_si;
    ux_e_mat.convertTo(norm_ue, CV_8U, e_scale, e_shift);
    uy_e_mat.convertTo(norm_ve, CV_8U, e_scale, e_shift);
    speed_e_mat.convertTo(norm_se, CV_8U, e_scale, e_shift);
    ux_i_mat.convertTo(norm_ui, CV_8U, i_scale, i_shift);
    uy_i_mat.convertTo(norm_vi, CV_8U, i_scale, i_shift);
    speed_i_mat.convertTo(norm_si, CV_8U, i_scale, i_shift);

    // ───────────────────────────────────────────────────────────────
    // 6) Apply JET colormap to each normalized mat
    // ───────────────────────────────────────────────────────────────
    cv::Mat heat_ue, heat_ve, heat_se;
    cv::Mat heat_ui, heat_vi, heat_si;
    cv::applyColorMap(norm_ue, heat_ue, cv::COLORMAP_JET);
    cv::applyColorMap(norm_ve, heat_ve, cv::COLORMAP_JET);
    cv::applyColorMap(norm_se, heat_se, cv::COLORMAP_JET);
    cv::applyColorMap(norm_ui, heat_ui, cv::COLORMAP_JET);
    cv::applyColorMap(norm_vi, heat_vi, cv::COLORMAP_JET);
    cv::applyColorMap(norm_si, heat_si, cv::COLORMAP_JET);

    // ───────────────────────────────────────────────────────────────
    // 7) Flip each vertically so that (0,0) is at bottom‐left
    // ───────────────────────────────────────────────────────────────
    cv::flip(heat_ue, heat_ue, 0);
    cv::flip(heat_ve, heat_ve, 0);
    cv::flip(heat_se, heat_se, 0);
    cv::flip(heat_ui, heat_ui, 0);
    cv::flip(heat_vi, heat_vi, 0);
    cv::flip(heat_si, heat_si, 0);

    // ───────────────────────────────────────────────────────────────
    // 8) Wrap each colored heatmap with its own white border + bottom label
    // ───────────────────────────────────────────────────────────────
    auto wrap_with_label = [&](const cv::Mat &src, const std::string &txt) {
        cv::Mat tmp;
        cv::copyMakeBorder(
            src, tmp,
            border, border + label_height,
            border, border,
            cv::BORDER_CONSTANT, cv::Scalar(255,255,255)
        );
        cv::putText(
            tmp,
            txt,
            cv::Point(border + 5, tmp.rows - 5),  // 5px from left, 5px above bottom
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0,0,0),
            1
        );
        return tmp;
    };

    cv::Mat w_ue = wrap_with_label(heat_ue, "u_x_e");    // you can write “u_x_e” if you prefer ASCII
    cv::Mat w_ve = wrap_with_label(heat_ve, "u_y_e");
    cv::Mat w_se = wrap_with_label(heat_se, "|u_e|");
    cv::Mat w_ui = wrap_with_label(heat_ui, "u_x_i");
    cv::Mat w_vi = wrap_with_label(heat_vi, "u_y_i");
    cv::Mat w_si = wrap_with_label(heat_si, "|u_i|");

    // ───────────────────────────────────────────────────────────────
    // 9) Build the 2×3 “grid”:
    //     ┌─────────────────────────────────────────────┐
    //     │ [ w_ue │ w_ve │ w_se ]   ← top row (electrons) │
    //     │ [ w_ui │ w_vi │ w_si ]   ← bot row (ions)     │
    //     └─────────────────────────────────────────────┘
    cv::Mat top_row, bottom_row, grid;
    cv::hconcat(std::vector<cv::Mat>{ w_ue, w_ve, w_se }, top_row);
    cv::hconcat(std::vector<cv::Mat>{ w_ui, w_vi, w_si }, bottom_row);
    cv::vconcat(top_row, bottom_row, grid);
    // Now grid.cols = 3×panel_w, grid.rows = 2×panel_h

    // ───────────────────────────────────────────────────────────────
    // 10) Build TWO **fixed‐range** legend panels:
    //   • Left panel uses (e_min,e_mid,e_max) = (–1.5, 0.0, +1.5)  for electrons
    //   • Right panel uses (i_min,i_mid,i_max) = (–0.5, 0.0, +0.5)  for ions
    // Each also has a one‐line note (“e range: ±1.5e–13”) or (“i range: ±0.5e–18”).
    // ───────────────────────────────────────────────────────────────

    // 10a) ELECTRON legend (leftmost)
    // Build a gray‐gradient strip of height = legend_h, width = 1, then apply JET
    cv::Mat e_gray(legend_h, 1, CV_8U);
    for (int i = 0; i < legend_h; ++i) {
        // from white (255) at top → black (0) at bottom
        e_gray.at<uchar>(i, 0) = static_cast<uchar>(255 - (i * 255 / (legend_h - 1)));
    }
    cv::Mat e_color;
    cv::applyColorMap(e_gray, e_color, cv::COLORMAP_JET);
    cv::resize(e_color, e_color, cv::Size(legend_strip_w, legend_h), 0, 0, cv::INTER_NEAREST);

    // Create a white background of size (legend_h + 2·border) × legend_panel_w:
    cv::Mat left_panel(legend_h + 2*border, legend_panel_w, CV_8UC3, cv::Scalar(255,255,255));
    // Paste the JET strip at (border, border):
    cv::Rect e_roi(border, border, legend_strip_w, legend_h);
    e_color.copyTo(left_panel(e_roi));

    // 10a-1) Write the one‐line note “e range: ±1.5e–13” in that top border area
    std::ostringstream oss_e_note;
    oss_e_note << "e: *10e-13:";
    cv::Scalar txtcol(0,0,0);
    int fnt = cv::FONT_HERSHEY_SIMPLEX;
    double fsc = 0.5;
    int thr = 1;
    int baseline;
    cv::Size ts_e_note = cv::getTextSize(oss_e_note.str(), fnt, fsc, thr, &baseline);
    // Place it so its baseline is about halfway down the top border:
    int x_e_note = border;
    int y_e_note = (border + ts_e_note.height) / 2;
    cv::putText(left_panel, oss_e_note.str(), cv::Point(x_e_note, y_e_note), fnt, fsc, txtcol, thr);

    // 10a-2) Draw numeric labels “+1.50 / 0.00 / -1.50” next to the strip:
    double e_mid = 0.0;
    std::ostringstream oss_e_max, oss_e_mid, oss_e_min;
    oss_e_max << std::fixed << std::setprecision(1) << e_max;  // “1.50”
    oss_e_mid << std::fixed << std::setprecision(1) << e_mid;  // “0.00”
    oss_e_min << std::fixed << std::setprecision(1) << e_min;  // “-1.50”

    // Measure text heights for aligning vertically
    cv::Size ts_e_max = cv::getTextSize(oss_e_max.str(), fnt, fsc, thr, &baseline);
    cv::Size ts_e_mid = cv::getTextSize(oss_e_mid.str(), fnt, fsc, thr, &baseline);
    // (we don’t need ts_e_min for positioning, since we put it flush at bottom)

    int x_text_e = border + legend_strip_w + 5;  // 5px gap to the right of the strip
    int y_text_e_max = border + ts_e_max.height; // just below top border
    int y_text_e_mid = border + (legend_h/2) + (ts_e_mid.height/2);
    int y_text_e_min = border + legend_h;       // flush at bottom of the color‐bar

    cv::putText(left_panel, oss_e_max.str(), cv::Point(x_text_e, y_text_e_max), fnt, fsc, txtcol, thr);
    cv::putText(left_panel, oss_e_mid.str(), cv::Point(x_text_e, y_text_e_mid), fnt, fsc, txtcol, thr);
    cv::putText(left_panel, oss_e_min.str(), cv::Point(x_text_e, y_text_e_min), fnt, fsc, txtcol, thr);

    // 10b) ION legend (rightmost), exactly the same pattern:
    // Build the gray strip → JET (height = legend_h, width = legend_strip_w)
    cv::Mat i_gray(legend_h, 1, CV_8U);
    for (int i = 0; i < legend_h; ++i) {
        i_gray.at<uchar>(i,0) = static_cast<uchar>(255 - (i * 255 / (legend_h - 1)));
    }
    cv::Mat i_color;
    cv::applyColorMap(i_gray, i_color, cv::COLORMAP_JET);
    cv::resize(i_color, i_color, cv::Size(legend_strip_w, legend_h), 0, 0, cv::INTER_NEAREST);

    cv::Mat right_panel(legend_h + 2*border, legend_panel_w, CV_8UC3, cv::Scalar(255,255,255));
    cv::Rect i_roi(border, border, legend_strip_w, legend_h);
    i_color.copyTo(right_panel(i_roi));

    // 10b-1) One‐line note “i range: ±0.5e–18”
    std::ostringstream oss_i_note;
    oss_i_note << "i: *10e-18";
    cv::Size ts_i_note = cv::getTextSize(oss_i_note.str(), fnt, fsc, thr, &baseline);
    int x_i_note = border;
    int y_i_note = (border + ts_i_note.height) / 2;
    cv::putText(right_panel, oss_i_note.str(), cv::Point(x_i_note, y_i_note), fnt, fsc, txtcol, thr);

    // 10b-2) Numeric labels “+0.50 / 0.00 / -0.50”
    double i_mid = 0.0;
    std::ostringstream oss_i_max, oss_i_mid, oss_i_min;
    oss_i_max << std::fixed << std::setprecision(1) << i_max;   // “0.50”
    oss_i_mid << std::fixed << std::setprecision(1) << i_mid;   // “0.00”
    oss_i_min << std::fixed << std::setprecision(1) << i_min;   // “-0.50”

    cv::Size ts_i_max = cv::getTextSize(oss_i_max.str(), fnt, fsc, thr, &baseline);
    cv::Size ts_i_mid = cv::getTextSize(oss_i_mid.str(), fnt, fsc, thr, &baseline);

    int x_text_i = border + legend_strip_w + 5;
    int y_text_i_max = border + ts_i_max.height;
    int y_text_i_mid = border + (legend_h/2) + (ts_i_mid.height/2);
    int y_text_i_min = border + legend_h;

    cv::putText(right_panel, oss_i_max.str(), cv::Point(x_text_i, y_text_i_max), fnt, fsc, txtcol, thr);
    cv::putText(right_panel, oss_i_mid.str(), cv::Point(x_text_i, y_text_i_mid), fnt, fsc, txtcol, thr);
    cv::putText(right_panel, oss_i_min.str(), cv::Point(x_text_i, y_text_i_min), fnt, fsc, txtcol, thr);

    // ───────────────────────────────────────────────────────────────
    // 11) Combine left_panel + grid + right_panel into one final frame
    // ───────────────────────────────────────────────────────────────
    cv::Mat output_frame(frame_h, frame_w, CV_8UC3, cv::Scalar(255,255,255));

    // 11a) Paste left_panel at (border, border)
    left_panel.copyTo(output_frame(cv::Rect(
        border,
        border,
        legend_panel_w,
        legend_h + 2*border
    )));

    // 11b) Paste grid at (border + legend_panel_w + border, border)
    int grid_x = border + legend_panel_w + border;
    grid.copyTo(output_frame(cv::Rect(
        grid_x,
        border,
        grid_w,
        grid_h
    )));

    // 11c) Paste right_panel at (grid_x + grid_w + border, border)
    int ion_x = grid_x + grid_w + border;
    right_panel.copyTo(output_frame(cv::Rect(
        ion_x,
        border,
        legend_panel_w,
        legend_h + 2*border
    )));

    // ───────────────────────────────────────────────────────────────
    // 12) Write this final frame to the velocity‐video writer
    // ───────────────────────────────────────────────────────────────
    video_writer_velocity.write(output_frame);
}




// --------------------------------------------------------------
// Visualization for Temperature
// --------------------------------------------------------------

void LBmethod::VisualizationTemperature(size_t t) {
    constexpr int border       = 10;
    constexpr int label_height = 30;

    static cv::Mat temp_mat_e, temp_mat_i;
    static cv::Mat temp_heatmap_e, temp_heatmap_i;
    static cv::Mat wrapped_e, wrapped_i;
    static cv::Mat grid;
    static cv::Mat output_frame;

    // 1) Allocate on first call
    if (t == 0) {
        temp_mat_e = cv::Mat(NY, NX, CV_32F);
        temp_mat_i = cv::Mat(NY, NX, CV_32F);
    }

    // 2) Copy T_e, T_i
    #pragma omp parallel for collapse(2)
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
            size_t idx = INDEX(x, y);
            temp_mat_e.at<float>(y, x) = static_cast<float>(T_e[idx]);
            temp_mat_i.at<float>(y, x) = static_cast<float>(T_i[idx]);
        }
    }

    // 3) Global min/max across both
    double min_e, max_e, min_i, max_i;
    cv::minMaxLoc(temp_mat_e, &min_e, &max_e);
    cv::minMaxLoc(temp_mat_i, &min_i, &max_i);
    double frame_min = std::min(min_e, min_i);
    double frame_max = std::max(max_e, max_i);

    // 4) Update the running global extremes:
    temperature_min_global = std::min(temperature_min_global, frame_min);
    temperature_max_global = std::max(temperature_max_global, frame_max);

    // 5) From here on out, use density_min_global & density_max_global for normalization:
    double global_min = temperature_min_global;
    double global_max = temperature_max_global;

    // 4) Normalize to 8U with shared scale
    cv::Mat norm_e, norm_i;
    temp_mat_e.convertTo(norm_e, CV_8U,
                         255.0 / (global_max - global_min),
                         -255.0 * global_min / (global_max - global_min));
    temp_mat_i.convertTo(norm_i, CV_8U,
                         255.0 / (global_max - global_min),
                         -255.0 * global_min / (global_max - global_min));

    // 5) Apply JET
    cv::applyColorMap(norm_e, temp_heatmap_e, cv::COLORMAP_JET);
    cv::applyColorMap(norm_i, temp_heatmap_i, cv::COLORMAP_JET);

    // 6) Flip vertically
    cv::flip(temp_heatmap_e, temp_heatmap_e, 0);
    cv::flip(temp_heatmap_i, temp_heatmap_i, 0);

    // 7) Wrap with border & bottom text
    auto wrap_with_label = [&](const cv::Mat& src, const std::string& label_text) {
        cv::Mat bordered;
        cv::copyMakeBorder(
            src,
            bordered,
            border,
            border + label_height,
            border,
            border,
            cv::BORDER_CONSTANT,
            cv::Scalar(255,255,255)
        );
        cv::putText(
            bordered,
            label_text,
            cv::Point(border + 5, bordered.rows - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0,0,0),
            1
        );
        return bordered;
    };

    wrapped_e = wrap_with_label(temp_heatmap_e, "T_e");
    wrapped_i = wrap_with_label(temp_heatmap_i, "T_i");

    // 8) Concat two by side
    cv::hconcat(std::vector<cv::Mat>{wrapped_e, wrapped_i}, grid);

    // 9) Build vertical legend
    int legend_width  = 40;
    int legend_height = grid.rows - 2 * border;
    cv::Mat legend_gray(legend_height, 1, CV_8U);

    for (int i = 0; i < legend_height; ++i) {
        legend_gray.at<uchar>(i, 0) =
            static_cast<uchar>(255 - (i * 255 / (legend_height - 1)));
    }
    cv::Mat legend_color;
    cv::applyColorMap(legend_gray, legend_color, cv::COLORMAP_JET);
    cv::resize(legend_color, legend_color, cv::Size(legend_width, legend_height), 0, 0, cv::INTER_NEAREST);

    // 10) Legend panel with numeric labels
    int text_margin = 5;
    int panel_width = legend_width + 60;
    cv::Mat legend_panel(legend_height + 2 * border,
                         panel_width,
                         CV_8UC3,
                         cv::Scalar(255,255,255));
    cv::Rect legend_roi(border, border, legend_width, legend_height);
    legend_color.copyTo(legend_panel(legend_roi));

    // Draw text values
    cv::Scalar text_color(0,0,0);
    int font      = cv::FONT_HERSHEY_SIMPLEX;
    double fscale = 0.5;
    int thickness = 1;
    int baseline  = 0;

    std::ostringstream oss_max, oss_mid, oss_min;
    oss_max << std::fixed << std::setprecision(2) << global_max;
    oss_min << std::fixed << std::setprecision(2) << global_min;
    double midv = 0.5 * (global_min + global_max);
    oss_mid << std::fixed << std::setprecision(2) << midv;

    cv::Size ts_max = cv::getTextSize(oss_max.str(), font, fscale, thickness, &baseline);
    cv::Size ts_mid = cv::getTextSize(oss_mid.str(), font, fscale, thickness, &baseline);
    cv::Size ts_min = cv::getTextSize(oss_min.str(), font, fscale, thickness, &baseline);

    int x_text = border + legend_width + text_margin;
    int y_max  = border + ts_max.height;
    int y_mid  = border + (legend_height / 2) + (ts_mid.height / 2);
    int y_min  = border + legend_height - (ts_min.height / 2);

    cv::putText(legend_panel, oss_max.str(), cv::Point(x_text, y_max), font, fscale, text_color, thickness);
    cv::putText(legend_panel, oss_mid.str(), cv::Point(x_text, y_mid), font, fscale, text_color, thickness);
    cv::putText(legend_panel, oss_min.str(), cv::Point(x_text, y_min), font, fscale, text_color, thickness);

    // 11) Combine grid + legend_panel
    int total_height = grid.rows + 2 * border;
    int total_width  = grid.cols + panel_width + 3 * border;
    output_frame = cv::Mat(total_height, total_width, CV_8UC3, cv::Scalar(255, 255, 255));

    grid.copyTo(output_frame(cv::Rect(border, border, grid.cols, grid.rows)));
    legend_panel.copyTo(output_frame(cv::Rect(grid.cols + 2 * border,
                                              border,
                                              legend_panel.cols,
                                              legend_panel.rows)));

    // 12) Write to the temperature‐video writer
    video_writer_temperature.write(output_frame);
}
