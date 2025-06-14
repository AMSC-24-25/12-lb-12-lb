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
                   const double    _Ex_SI,
                   const double    _Ey_SI,
                   const double    _T_e_SI_init,
                   const double    _T_i_SI_init,
                   const double    _n_0_SI_init,
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
      Ex_SI      (_Ex_SI),
      Ey_SI      (_Ey_SI),
      T_e_SI_init(_T_e_SI_init),
      T_i_SI_init(_T_i_SI_init),
      n_0_SI_init(_n_0_SI_init),
      poisson_type(_poisson_type),
      bc_type     (_bc_type),
      omega_sor   (_omega_sor)
{   //The rescaling has already been done in the hpp

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

    rho_e.assign(NX * NY, 0.0);
    rho_i.assign(NX * NY, 0.0);
    ux_e.assign(NX * NY, 0.0);
    uy_e.assign(NX * NY, 0.0);
    ux_i.assign(NX * NY, 0.0);
    uy_i.assign(NX * NY, 0.0);
    ux_e_i.assign(NX * NY, 0.0);
    uy_e_i.assign(NX * NY, 0.0);

    T_e.assign(NX * NY, 0.0);
    T_i.assign(NX * NY, 0.0);

    phi.assign(NX * NY, 0.0);
    phi_new.assign(NX * NY, 0.0);
    Ex.assign(NX * NY, Ex_ext);
    Ey.assign(NX * NY, Ey_ext);

    // In lattice units, store ρ_q_latt = (ρ_i - ρ_e) * 1.0  (just #/cell difference)
    rho_q.assign(NX * NY, 0.0); //Should be rho_q_latt[idx] = (n_i[idx] - n_e[idx]) * q_0;

    // 9) Initialize fields:  set f = f_eq(rho=1,u=0), zero φ, E=Ex_latt_init
    Initialize();
    // 10) Print initial values to check
    std::cout
        << "LBmethod initialized with:\n"
        << "NX = " << NX << ", NY = " << NY << ", NSTEPS = " << NSTEPS << "\n"
        << "Z_ion = " << Z_ion << ", A_ion = " << A_ion << ", r_ion = " << r_ion << " m\n"
        << "poisson_type = " << static_cast<int>(poisson_type) << ", bc_type = " << static_cast<int>(bc_type) << "\n"
        << "omega_sor = " << omega_sor << "\n"
        << "Chosen quantity to relate are:\n"
        << "M0 = " << M0_SI << " (kg), Q0 = " << Q0_SI << " (C), T0 = " << T0_SI << " (K)\n"
        << "L0 = " << L0_SI << " (m), t0 = " << t0_SI << " (s)\n"
        << "With that we have:\n"
        << "cs2 = " << cs2 << " (lattice unit)\n"
        << "Ex_ext = " << Ex_ext << " (lattice unit), Ey_ext = " << Ey_ext << " (lattice unit)\n"
        << "T_e_init = " << T_e_init << " (lattice unit), T_i_init = " << T_i_init << " (lattice unit)\n"
        << "and imposed:\n"
        << "tau_e = " << tau_e << ", tau_i = " << tau_i << ", tau_e_i = " << tau_e_i << "\n"
        << "tau_Te = " << tau_Te << ", tau_Ti = " << tau_Ti << ", tau_Te_Ti = " << tau_Te_Ti << "\n"
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
    for (size_t y = 1*NY/4; y < 3*NY/4; ++y) {
            for (size_t x = 1*NY/4; x < 3*NX/4; ++x) {
                const size_t idx = INDEX(x, y);
                rho_e[idx] = rho_e_init; // Set initial electron density
                T_e[idx] = T_e_init; // Set initial electron temperature
                for (size_t i = 0; i < Q; ++i) {
                    const size_t idx_3 = INDEX(x, y, i);
                    f_e[idx_3] = f_eq_e[idx_3] = w[i] * m_e; // Equilibrium function for electrons
                    g_e[idx_3] = g_eq_e[idx_3] = w[i] * T_e_init; // Thermal function for electrons
                    f_eq_e_i[idx_3] = w[i] * m_e; // Equilibrium function for electron-ion interaction
                }
            }
            for (size_t x = 1*NY/4; x < 3*NX/4; ++x) {
                const size_t idx = INDEX(x, y);
                rho_i[idx] = rho_i_init; // Set initial ion density
                T_i[idx] = T_i_init; // Set initial ion temperature
                for (size_t i = 0; i < Q; ++i) {
                    const size_t idx_3 = INDEX(x, y, i);
                    f_i[idx_3] = f_eq_i[idx_3] = w[i] * m_i; // Equilibrium function for ions
                    g_i[idx_3] = g_eq_i[idx_3] = w[i] * T_i_init; // Thermal function for ions
                    f_eq_i_e[idx_3] = w[i] * m_i; // Equilibrium function for ion-electron interaction
                }
            }
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
                const double u2_e_i = ux_e_i[idx] * ux_e_i[idx] + uy_e_i[idx] * uy_e_i[idx]; // Square of the speed magnitude for electron-ion interaction
                const double den_e = rho_e[idx]; // Electron density
                const double den_i = rho_i[idx]; // Ion density
                
                for (size_t i = 0; i < Q; ++i) {
                    const size_t idx_3=INDEX(x, y, i);
                    const double cu_e = cx[i]*ux_e[idx] +cy[i]*uy_e[idx]; // Dot product (c_i · u)
                    const double cu_i = cx[i]*ux_i[idx] +cy[i]*uy_i[idx];
                    const double cu_e_i = cx[i]*ux_e_i[idx] +cy[i]*uy_e_i[idx]; // Dot product (c_i · u) for electron-ion interaction
                    
                    // Compute f_eq from discretization of Maxwell Boltzmann distribution function
                    f_eq_e[idx_3]= w[i]*den_e*(
                        1.0 +
                        (cu_e / cs2) +
                        (cu_e * cu_e) / (2.0 * cs2 * cs2) -
                        u2_e / (2.0 * cs2)
                    );
                    f_eq_i[idx_3]= w[i]*den_i*(
                        1.0 +
                        (cu_i / cs2) +
                        (cu_i * cu_i) / (2.0 * cs2 * cs2) -
                        u2_i / (2.0 * cs2)
                    );

                    f_eq_e_i[idx_3]= w[i]*den_e*(
                        1.0 +
                        (cu_e_i / cs2) +
                        (cu_e_i * cu_e_i) / (2.0 * cs2 * cs2) -
                        u2_e_i / (2.0 * cs2)
                    );
                    f_eq_i_e[idx_3]= w[i]*den_i*(
                        1.0 +
                        (cu_e_i / cs2) +
                        (cu_e_i * cu_e_i) / (2.0 * cs2 * cs2) -
                        u2_e_i / (2.0 * cs2)
                    );

                    //still yet to implement
                    g_eq_e[idx_3]=w[i]*T_e[idx]*(
                        1.0 +
                        (cu_e / cs2) +
                        (cu_e * cu_e ) / (2.0 * cs2 *cs2) -
                        u2_e / (2.0 * cs2)
                    );
                    g_eq_i[idx_3]=w[i]*T_i[idx]*(
                        1.0 +
                        (cu_i / cs2) +
                        (cu_i * cu_i ) / (2.0 * cs2 *cs2) -
                        u2_i / (2.0 * cs2)
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
    double rho_loc_e = 0.0;
    double ux_loc_e = 0.0;
    double uy_loc_e = 0.0;
    double T_loc_e = 0.0;

    double rho_loc_i = 0.0;
    double ux_loc_i = 0.0;
    double uy_loc_i = 0.0;
    double T_loc_i = 0.0;

    //ideas to calculate the relaxation parameters from physical properties of collisions
    //const double sigma_el_ion=M_PI*(r_el+r_ion)*(r_el+r_ion); //Cross section for electron-ion interaction
    //const double mean_vel_el = std::sqrt(8*kb/(m_el*M_PI)); //Mean velocity of electrons without Temperature dependence
    //const double mean_vel_ion = std::sqrt(8*kb/(A_ion*m_p*M_PI)); //Mean velocity of ions without Temperature dependence
    //const double mass_el_ion = m_el * A_ion * m_p /(m_el + A_ion * m_p); //Reduced mass for electron-ion interaction
    #pragma omp parallel for collapse(2) private(rho_loc_e, ux_loc_e, uy_loc_e, T_loc_e, rho_loc_i, ux_loc_i, uy_loc_i, T_loc_i)
        for (size_t x=0; x<NX; ++x){
            for (size_t y = 0; y < NY; ++y) {
                const size_t idx = INDEX(x, y);
                rho_loc_e = 0.0;
                ux_loc_e = 0.0;
                uy_loc_e = 0.0;
                T_loc_e = 0.0;

                rho_loc_i = 0.0;
                ux_loc_i = 0.0;
                uy_loc_i = 0.0;
                T_loc_i = 0.0;


                for (size_t i = 0; i < Q; ++i) {
                    const size_t idx_3 = INDEX(x, y, i);
                    const double fi_e=f_e[idx_3];
                    rho_loc_e += fi_e;
                    ux_loc_e += fi_e * cx[i];
                    uy_loc_e += fi_e * cy[i];
                    T_loc_e += g_e[idx_3];

                    const double fi_i=f_i[idx_3];
                    rho_loc_i += fi_i;
                    ux_loc_i += fi_i * cx[i];
                    uy_loc_i += fi_i * cy[i];
                    T_loc_i += g_i[idx_3];
                    
                }
                if (rho_loc_e<1e-10){//in order to avoid instabilities (can be removed if the code is stable)
                    rho_e[idx] = 0.0;
                    ux_e[idx] = 0.0;
                    uy_e[idx] = 0.0;
                    T_e[idx] = 0.0;
                }else if (rho_loc_i<1e-10){
                    rho_i[idx] = 0.0;
                    ux_i[idx] = 0.0;
                    uy_i[idx] = 0.0;
                    T_i[idx] = 0.0;
                }
                else {
                    rho_e[idx] = rho_loc_e;
                    rho_i[idx] = rho_loc_i;

                    // Assign velocities
                    ux_e[idx] = ux_loc_e / rho_loc_e;
                    uy_e[idx] = uy_loc_e / rho_loc_e;
                    ux_i[idx] = ux_loc_i / rho_loc_i;
                    uy_i[idx] = uy_loc_i / rho_loc_i;

                    //Add the force term to velocity
                    ux_e[idx] += 0.5 * q_e * Ex[idx] / m_e;
                    uy_e[idx] += 0.5 * q_e * Ey[idx] / m_e;
                    ux_i[idx] += 0.5 * q_i * Ex[idx] / m_i;
                    uy_i[idx] += 0.5 * q_i * Ey[idx] / m_i;

                    // Assign Temperature
                    T_e[idx] = T_loc_e;
                    T_i[idx] = T_loc_i; 

                    /////Other parameters not yet implemented
                    //Relaxation times
                    //tau_el[idx] = 1.0/(M_PI*(2*r_el)*(2*r_el) * n_local_el * mean_vel_el * std::sqrt(T_el[idx])); //Relaxation time for electron
                    //tau_ion[idx] = 1.0/(M_PI*(2*r_ion)*(2*r_ion) * n_local_ion * mean_vel_ion * std::sqrt(T_ion[idx])); //Relaxation time for ion
    
                    //Interaction velocities
                    ux_e_i[idx] = (rho_loc_e * ux_e[idx] + rho_loc_i * ux_i[idx]) / (rho_loc_e + rho_loc_i);
                    uy_e_i[idx] = (rho_loc_e * uy_e[idx] + rho_loc_i * uy_i[idx]) / (rho_loc_e + rho_loc_i);

                    //Interaction relaxation times
                    //tau_el_ion[idx] = 1.0/(sigma_el_ion * n_local_ion * mean_vel_el * std::sqrt(T_el[idx]));
                    //tau_ion_el[idx] = 1.0/(sigma_el_ion * n_local_el * mean_vel_ion * std::sqrt(T_ion[idx]));
                }
                // Lattice‐unit charge density (#/cell difference):
                rho_q[idx] = (q_i * rho_i[idx] / m_i + q_e * rho_e[idx] / m_e);
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
    // else if (poisson_type == PoissonType::NONE)    // No Poisson solver, use initial Ex, Ey
}
//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: Gauss–Seidel with Dirichlet φ=0 on boundary.
//    We solve  ∇² φ = − ρ_q_phys / ε₀,  in lattice units.
//    Our RHS in “lattice‐Land” is:  RHS_latt = − (ρ_q_phys/ε₀) * dx_SI * dx_SI.
//    Then φ_new[i,j] = ¼ [ φ[i+1,j] + φ[i−1,j] + φ[i,j+1] + φ[i,j−1] − RHS_latt[i,j] ].
//
//  After convergence, we reconstruct E with centered differences:
//    E_x = −(φ[i+1,j] − φ[i−1,j]) / (2 * dx_SI), etc.
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::SolvePoisson_GS() {
    std::vector<double> RHS(NX*NY, 0.0);

    // Build RHS_latt = −(ρ_q_phys / ε₀) * dx_SI
    for(size_t idx=0; idx<NX*NY; ++idx) {
        RHS[idx] = - rho_q[idx];
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
            Ex[idx] = - (phi[INDEX(i+1,j)] - phi[INDEX(i-1,j)]) / (2.0);
            Ey[idx] = - (phi[INDEX(i,j+1)] - phi[INDEX(i,j-1)]) / (2.0);
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
                double phi_GS = 0.25 * (nbSum + rho_q[idx]);
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
            Ex[idx] = - (phi[INDEX(i+1,j)] - phi[INDEX(i-1,j)]) / (2.0);
            Ey[idx] = - (phi[INDEX(i,j+1)] - phi[INDEX(i,j-1)]) / (2.0);
        }
    }
    for(size_t i=0; i<NX; ++i) {
        Ex[INDEX(i,0)]     = 0;
        Ey[INDEX(i,0)]     = 0;
        Ex[INDEX(i,NY-1)]  = 0;
        Ey[INDEX(i,NY-1)]  = 0;
    }
    for(size_t j=0; j<NY; ++j) {
        Ex[INDEX(0,j)]     = 0;
        Ey[INDEX(0,j)]     = 0;
        Ex[INDEX(NX-1,j)]  = 0;
        Ey[INDEX(NX-1,j)]  = 0;
    }
}
void LBmethod::SolvePoisson_fft() {  ///huge revision needed for this one
    //Periodic conditions ONLY

    const double inveps = 1.0;

    // Allocate FFTW arrays
    fftw_complex *rho_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NX * NY);
    fftw_complex *phi_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NX * NY);

    fftw_plan forward_plan = fftw_plan_dft_r2c_2d(NX, NY, phi.data(), rho_hat, FFTW_ESTIMATE);
    fftw_plan backward_plan = fftw_plan_dft_c2r_2d(NX, NY, phi_hat, phi.data(), FFTW_ESTIMATE);


    // Copy rho_c into phi temporarily to use r2c FFT (input must be real)
    std::copy(rho_q.begin(), rho_q.end(), phi.begin());

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
        total_charge += rho_q[idx];
    }
    total_charge=total_charge*e_charge_SI; // Convert to physical units
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
                
                const double F_e = w[i] * q_e * rho_e[idx] / m_e / cs2 * (1.0-1.0/(2*tau_e)) * (
                    (cx[i]*Ex_loc+cy[i]*Ey_loc)+
                    (cx[i]*ux_e[idx]+cy[i]*uy_e[idx])*(cx[i]*Ex_loc+cy[i]*Ey_loc)/cs2-
                    (ux_e[idx]*Ex_loc+uy_e[idx]*Ey_loc)
                );
                const double F_i = w[i] * q_i * rho_i[idx] / m_i / cs2 * (1.0-1.0/(2*tau_i)) * (
                    (cx[i]*Ex_loc+cy[i]*Ey_loc)+
                    (cx[i]*ux_i[idx]+cy[i]*uy_i[idx])*(cx[i]*Ex_loc+cy[i]*Ey_loc)/cs2-
                    (ux_i[idx]*Ex_loc+uy_i[idx]*Ey_loc)
                );//maybe simplify a bit the writing
               
                // Compute complete collisions terms
                const double C_e = -(f_e[idx_3]-f_eq_e[idx_3]) / tau_e -(f_e[idx_3]-f_eq_e_i[idx_3]) / tau_e_i;
                const double C_i = -(f_i[idx_3]-f_eq_i[idx_3]) / tau_i -(f_i[idx_3]-f_eq_i_e[idx_3]) / tau_e_i;

                // Update distribution functions with Guo forcing term
                f_temp_e[idx_3] = f_e[idx_3] + C_e + F_e;
                f_temp_i[idx_3] = f_i[idx_3] + C_i + F_i;
            }
        }
    }
    // Swap temporary arrays with main arrays
    f_e.swap(f_temp_e);
    f_i.swap(f_temp_i);
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
                const double C_e = -(g_e[idx_3]-g_eq_e[idx_3]) / tau_Te;
                const double C_i = -(g_i[idx_3]-g_eq_i[idx_3]) / tau_Ti;

                // Update distribution functions with Guo forcing term
                g_temp_e[idx_3] = f_e[idx_3] + C_e;
                g_temp_i[idx_3] = f_i[idx_3] + C_i;
            }
        }
    }
    // Swap temporary arrays with main arrays
    g_e.swap(g_temp_e);
    g_i.swap(g_temp_i);
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
                    else if(x_str >= 0 && x_str <static_cast<int>(NX)){
                        //We are outside so bounceback
                        f_temp_e[INDEX(x_str,y,opp[i])]=f_e[INDEX(x,y,i)];
                        f_temp_i[INDEX(x_str,y,opp[i])]=f_i[INDEX(x,y,i)];
                    }
                    else if(y_str >= 0 && y_str <static_cast<int>(NY)){
                        //We are outside so bounceback
                        f_temp_e[INDEX(x,y_str,opp[i])]=f_e[INDEX(x,y,i)];
                        f_temp_i[INDEX(x,y_str,opp[i])]=f_i[INDEX(x,y,i)];
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
    const double fps       = 10.0; // frames per second for videos

    // --- Density‐video (2 panels side by side) ---
    {
        int legend_width = 40, text_area = 60;
        int panel_width  = legend_width + text_area;
        int frame_w = 3 * tile_w + 2 * panel_width + 4 * border;
        int frame_h = tile_h;
        video_writer_density.open(
            "density_video.mp4",
            cv::VideoWriter::fourcc('m','p','4','v'),
            fps,                        // fps
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
        int frame_w = 3 * (NX + 2 * border);                         // 3 tiles in larghezza
        int frame_h = 2 * (NY + 2 * border + label_height);          // 2 tiles in altezza

        video_writer_velocity.open(
            "velocity_video.mp4",
            cv::VideoWriter::fourcc('m','p','4','v'),
            fps,
            cv::Size(frame_w, frame_h),
            true
        );
        if (!video_writer_velocity.isOpened()) {
            std::cerr << "Cannot open velocity_video.mp4 for writing\n";
            return;
        }
    }



    // --- Temperature‐video (2 panels side by side) ---
    {
        int frame_w = 2 * tile_w;
        int frame_h = tile_h;
        video_writer_temperature.open(
            "temperature_video.mp4",
            cv::VideoWriter::fourcc('m','p','4','v'),
            fps,
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
        if (t%1==0) {
            auto max_rho_e = std::max_element(rho_e.begin(), rho_e.end());
            auto min_rho_e = std::min_element(rho_e.begin(), rho_e.end());
            auto max_rho_i = std::max_element(rho_i.begin(), rho_i.end());
            auto min_rho_i = std::min_element(rho_i.begin(), rho_i.end());
            auto max_uxe = std::max_element(ux_e.begin(), ux_e.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto min_uxe = std::min_element(ux_e.begin(), ux_e.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto max_uye = std::max_element(uy_e.begin(), uy_e.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto min_uye = std::min_element(uy_e.begin(), uy_e.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto max_uxi = std::max_element(ux_i.begin(), ux_i.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto min_uxi = std::min_element(ux_i.begin(), ux_i.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto max_uyi = std::max_element(uy_i.begin(), uy_i.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto min_uyi = std::min_element(uy_i.begin(), uy_i.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto max_Ex = std::max_element(Ex.begin(), Ex.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto min_Ex = std::min_element(Ex.begin(), Ex.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto max_Ey = std::max_element(Ey.begin(), Ey.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto min_Ey = std::min_element(Ey.begin(), Ey.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto max_rho = std::max_element(rho_q.begin(), rho_q.end());
            auto min_rho = std::min_element(rho_q.begin(), rho_q.end());
            auto max_Te = std::max_element(T_e.begin(), T_e.end());
            auto min_Te = std::min_element(T_e.begin(), T_e.end());
            auto max_Ti = std::max_element(T_i.begin(), T_i.end());
            auto min_Ti = std::min_element(T_i.begin(), T_i.end());
            std::cout <<"Step:"<<t<<std::endl;
            std::cout <<"max rho_e= "<<*max_rho_e<<", min rho_e= "<<*min_rho_e<<std::endl;
            std::cout <<"max rho_i= "<<*max_rho_i<<", min rho_i= "<<*min_rho_i<<std::endl;
            std::cout <<"max ux_e= "<<*max_uxe<<", min ux_e= "<<*min_uxe<<std::endl;
            std::cout <<"max ux_i= "<<*max_uxi<<", min ux_i= "<<*min_uxi<<std::endl;
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
                    std::cout<<rho_e[INDEX(x,y)]<< " ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
            std::cout<<"n_i= "<<std::endl;
            for (size_t x = 0; x < NX; ++x) {
                for (size_t y = 0; y < NY; ++y) {
                    std::cout<<rho_i[INDEX(x,y)]<< " ";
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
            std::cout<<"rho= "<<std::endl;
            for (size_t x = 0; x < NX; ++x) {
                for (size_t y = 0; y < NY; ++y) {
                    std::cout<<rho_q[INDEX(x,y)]<< " ";
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
        Collisions(); // f(x,y,t+1)=f(x-cx,y-cy,t) + tau * (f_eq - f) + dt*F
        ThermalCollisions();
        Streaming(); // f(x,y,t+1)=f(x-cx,y-cy,t)
        ThermalStreaming_BounceBack();
        SolvePoisson();

        VisualizationDensity();
        VisualizationVelocity();
        VisualizationTemperature();
    }

    video_writer_density.release();
    video_writer_velocity.release();
    video_writer_temperature.release();
    std::cout << "Video saved, simulation ended " << std::endl;
}
//──────────────────────────────────────────────────────────────────────────────
//  Visualization stub.  Use OpenCV to save density images at time t, etc.
//──────────────────────────────────────────────────────────────────────────────

// ============================
// DENSITY VISUALIZATION
// ============================
void LBmethod::VisualizationDensity() {
    constexpr int border = 10;
    constexpr int label_height = 30;

    static cv::Mat mat_n_e(NY, NX, CV_32F);
    static cv::Mat mat_n_i(NY, NX, CV_32F);
    static cv::Mat mat_rho_q(NY, NX, CV_32F);

    #pragma omp parallel for collapse(2)
    for (int x = 0; x < static_cast<int>(NX); ++x) {
        for (int y = 0; y < static_cast<int>(NY); ++y) {
            size_t idx = INDEX(x, y);
            mat_n_e.at<float>(y, x) = static_cast<float>(rho_e[idx]);
            mat_n_i.at<float>(y, x) = static_cast<float>(rho_i[idx]);
            mat_rho_q.at<float>(y, x) = static_cast<float>(rho_q[idx]);
        }
    }

    auto normalize_and_color = [](const cv::Mat& src, double vmin, double vmax) {
        cv::Mat norm, color;
        src.convertTo(norm, CV_8U, 255.0 / (vmax - vmin), -vmin * 255.0 / (vmax - vmin));
        cv::applyColorMap(norm, color, cv::COLORMAP_JET);
        cv::flip(color, color, 0);
        return color;
    };

    auto wrap_with_label = [&](const cv::Mat& img, const std::string& label) {
        cv::Mat bordered;
        cv::copyMakeBorder(img, bordered, border, border + label_height, border, border,
                           cv::BORDER_CONSTANT, cv::Scalar(255,255,255));
        cv::putText(bordered, label, cv::Point(border + 5, bordered.rows - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 1);
        return bordered;
    };

    auto c_n_e = normalize_and_color(mat_n_e, DENSITY_MIN, DENSITY_MAX);
    auto c_rho = normalize_and_color(mat_rho_q, CHARGE_MIN, CHARGE_MAX);
    auto c_n_i = normalize_and_color(mat_n_i, DENSITY_MIN, DENSITY_MAX);

    auto w_n_e = wrap_with_label(c_n_e, "rho_e");
    auto w_rho = wrap_with_label(c_rho, "rho_q");
    auto w_n_i = wrap_with_label(c_n_i, "rho_i");

    cv::Mat grid;
    cv::hconcat(std::vector<cv::Mat>{w_n_e, w_rho, w_n_i}, grid);

    int total_width = grid.cols + 2 * (40 + 60) + 4 * border;
    int total_height = grid.rows;
    cv::Mat frame(total_height, total_width, CV_8UC3, cv::Scalar(255,255,255));

    // TODO: add legends on left and right if desired
    grid.copyTo(frame(cv::Rect((total_width - grid.cols) / 2, 0, grid.cols, grid.rows)));

    video_writer_density.write(frame);
}


// ============================
// VELOCITY VISUALIZATION
// ============================
void LBmethod::VisualizationVelocity() {
    constexpr int border = 10;
    constexpr int label_height = 30;

    static cv::Mat ux_e_mat(NY, NX, CV_32F), uy_e_mat(NY, NX, CV_32F), ue_mag(NY, NX, CV_32F);
    static cv::Mat ux_i_mat(NY, NX, CV_32F), uy_i_mat(NY, NX, CV_32F), ui_mag(NY, NX, CV_32F);

    #pragma omp parallel for collapse(2)
    for (int x = 0; x < static_cast<int>(NX); ++x) {
        for (int y = 0; y < static_cast<int>(NY); ++y) {
            size_t idx = INDEX(x, y);
            double ux_el = ux_e[idx], uy_el = uy_e[idx];
            double ux_ion = ux_i[idx], uy_ion = uy_i[idx];
            ux_e_mat.at<float>(y, x) = static_cast<float>(ux_el);
            uy_e_mat.at<float>(y, x) = static_cast<float>(uy_el);
            ue_mag.at<float>(y, x)   = static_cast<float>(std::sqrt(ux_el*ux_el + uy_el*uy_el));
            ux_i_mat.at<float>(y, x) = static_cast<float>(ux_ion);
            uy_i_mat.at<float>(y, x) = static_cast<float>(uy_ion);
            ui_mag.at<float>(y, x)   = static_cast<float>(std::sqrt(ux_ion*ux_ion + uy_ion*uy_ion));
        }
    }

    auto normalize_and_color = [](const cv::Mat& src, double vmin, double vmax) {
        cv::Mat norm, color;
        src.convertTo(norm, CV_8U, 255.0 / (vmax - vmin), -vmin * 255.0 / (vmax - vmin));
        cv::applyColorMap(norm, color, cv::COLORMAP_JET);
        cv::flip(color, color, 0);
        return color;
    };

    auto wrap_with_label = [&](const cv::Mat& img, const std::string& label) {
        cv::Mat bordered;
        cv::copyMakeBorder(img, bordered, border, border + label_height, border, border,
                           cv::BORDER_CONSTANT, cv::Scalar(255,255,255));
        cv::putText(bordered, label, cv::Point(border + 5, bordered.rows - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 1);
        return bordered;
    };

    auto ue_x = normalize_and_color(ux_e_mat, UX_E_MIN, UX_E_MAX);
    auto ue_y = normalize_and_color(uy_e_mat, UY_E_MIN, UY_E_MAX);
    auto ue_m = normalize_and_color(ue_mag, UE_MAG_MIN, UE_MAG_MAX);
    auto ui_x = normalize_and_color(ux_i_mat, UX_I_MIN, UX_I_MAX);
    auto ui_y = normalize_and_color(uy_i_mat, UY_I_MIN, UY_I_MAX);
    auto ui_m = normalize_and_color(ui_mag, UI_MAG_MIN, UI_MAG_MAX);

    auto w1 = wrap_with_label(ue_x, "ux_e");
    auto w2 = wrap_with_label(ue_y, "uy_e");
    auto w3 = wrap_with_label(ue_m, "|u_e|");
    auto w4 = wrap_with_label(ui_x, "ux_i");
    auto w5 = wrap_with_label(ui_y, "uy_i");
    auto w6 = wrap_with_label(ui_m, "|u_i|");

    cv::Mat top, bot, grid;
    cv::hconcat(std::vector<cv::Mat>{w1, w2, w3}, top);
    cv::hconcat(std::vector<cv::Mat>{w4, w5, w6}, bot);
    cv::vconcat(top, bot, grid);

    video_writer_velocity.write(grid);
}


// ============================
// TEMPERATURE VISUALIZATION
// ============================
void LBmethod::VisualizationTemperature() {
    constexpr int border = 10;
    constexpr int label_height = 30;

    static cv::Mat Te_mat(NY, NX, CV_32F), Ti_mat(NY, NX, CV_32F);

    #pragma omp parallel for collapse(2)
    for (int x = 0; x < static_cast<int>(NX); ++x) {
        for (int y = 0; y < static_cast<int>(NY); ++y) {
            size_t idx = INDEX(x, y);
            Te_mat.at<float>(y, x) = static_cast<float>(T_e[idx]);
            Ti_mat.at<float>(y, x) = static_cast<float>(T_i[idx]);
        }
    }

    auto normalize_and_color = [](const cv::Mat& src, double vmin, double vmax) {
        cv::Mat norm, color;
        src.convertTo(norm, CV_8U, 255.0 / (vmax - vmin), -vmin * 255.0 / (vmax - vmin));
        cv::applyColorMap(norm, color, cv::COLORMAP_JET);
        cv::flip(color, color, 0);
        return color;
    };

    auto wrap_with_label = [&](const cv::Mat& img, const std::string& label) {
        cv::Mat bordered;
        cv::copyMakeBorder(img, bordered, border, border + label_height, border, border,
                           cv::BORDER_CONSTANT, cv::Scalar(255,255,255));
        cv::putText(bordered, label, cv::Point(border + 5, bordered.rows - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 1);
        return bordered;
    };

    auto col_Te = normalize_and_color(Te_mat, TEMP_E_MIN, TEMP_E_MAX);
    auto col_Ti = normalize_and_color(Ti_mat, TEMP_I_MIN, TEMP_I_MAX);

    auto w_Te = wrap_with_label(col_Te, "T_e");
    auto w_Ti = wrap_with_label(col_Ti, "T_i");

    cv::Mat grid;
    cv::hconcat(std::vector<cv::Mat>{w_Te, w_Ti}, grid);

    video_writer_temperature.write(grid);
}