#include "plasma_debug.hpp"
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
                   const double    _Ex_SI,
                   const double    _Ey_SI,
                   const double    _T_e_SI_init,
                   const double    _T_i_SI_init,
                   const double    _T_n_SI_init,
                   const double    _n_e_SI_init,
                   const double    _n_n_SI_init,
                   const PoissonType _poisson_type,
                   const BCType      _bc_type,
                   const double    _omega_sor)
    : NSTEPS      (_NSTEPS),
      NX          (_NX),
      NY          (_NY),
      n_cores     (_n_cores),
      Z_ion       (_Z_ion),
      A_ion       (_A_ion),
      Ex_SI      (_Ex_SI),
      Ey_SI      (_Ey_SI),
      T_e_SI_init(_T_e_SI_init),
      T_i_SI_init(_T_i_SI_init),
      T_n_SI_init(_T_n_SI_init),
      n_e_SI_init(_n_e_SI_init),
      n_n_SI_init(_n_n_SI_init),
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
    f_n.assign(NX * NY * Q, 0.0);
    f_temp_n.assign(NX * NY * Q, 0.0);
    f_eq_n.assign(NX * NY * Q, 0.0);
    
    f_eq_e_i.assign(NX * NY * Q, 0.0);
    f_eq_i_e.assign(NX * NY * Q, 0.0);
    f_eq_e_n.assign(NX * NY * Q, 0.0);
    f_eq_n_e.assign(NX * NY * Q, 0.0);
    f_eq_i_n.assign(NX * NY * Q, 0.0);
    f_eq_n_i.assign(NX * NY * Q, 0.0);

    g_e.assign(NX * NY * Q, 0.0);
    g_i.assign(NX * NY * Q, 0.0);
    g_n.assign(NX * NY * Q, 0.0);
    g_eq_e.assign(NX * NY * Q, 0.0);
    g_eq_i.assign(NX * NY * Q, 0.0);
    g_eq_n.assign(NX * NY * Q, 0.0);
    g_temp_e.assign(NX * NY * Q, 0.0);
    g_temp_i.assign(NX * NY * Q, 0.0);
    g_temp_n.assign(NX * NY * Q, 0.0);

    g_eq_e_i.assign(NX * NY * Q, 0.0);
    g_eq_i_e.assign(NX * NY * Q, 0.0);
    g_eq_e_n.assign(NX * NY * Q, 0.0);
    g_eq_n_e.assign(NX * NY * Q, 0.0);
    g_eq_i_n.assign(NX * NY * Q, 0.0);
    g_eq_n_i.assign(NX * NY * Q, 0.0);

    rho_e.assign(NX * NY, 0.0);
    rho_i.assign(NX * NY, 0.0);
    rho_n.assign(NX * NY, 0.0);
    ux_e.assign(NX * NY, 0.0);
    uy_e.assign(NX * NY, 0.0);
    ux_i.assign(NX * NY, 0.0);
    uy_i.assign(NX * NY, 0.0);
    ux_n.assign(NX * NY, 0.0);
    uy_n.assign(NX * NY, 0.0);
    ux_e_i.assign(NX * NY, 0.0);
    uy_e_i.assign(NX * NY, 0.0);
    ux_e_n.assign(NX * NY, 0.0);
    uy_e_n.assign(NX * NY, 0.0);
    ux_i_n.assign(NX * NY, 0.0);
    uy_i_n.assign(NX * NY, 0.0);

    T_e.assign(NX * NY, 0.0);
    T_i.assign(NX * NY, 0.0);
    T_n.assign(NX * NY, 0.0);

    phi.assign(NX * NY, 0.0);
    phi_new.assign(NX * NY, 0.0);
    Ex.assign(NX * NY, Ex_ext);
    Ey.assign(NX * NY, Ey_ext);

    // In lattice units, store ρ_q_latt = (ρ_i - ρ_e) * 1.0  (just #/cell difference)
    rho_q.assign(NX * NY, 0.0); //Should be rho_q_latt[idx] = (n_i[idx] - n_e[idx]) * q_0;

    // 9) Initialize fields:  set f = w * m
    Initialize();
    // 10) Print initial values to check
    std::cout
        << "LBmethod initialized with:\n"
        << "NX = " << NX << ", NY = " << NY << ", NSTEPS = " << NSTEPS << "\n"
        << "Z_ion = " << Z_ion << ", A_ion = " << A_ion << "\n"
        << "poisson_type = " << static_cast<int>(poisson_type) << ", bc_type = " << static_cast<int>(bc_type) << "\n"
        << "omega_sor = " << omega_sor << "\n"
        << "Chosen quantity to relate are:\n"
        << "M0 = " << M0_SI << " (kg), Q0 = " << Q0_SI << " (C), T0 = " << T0_SI << " (K)\n"
        << "L0 = " << L0_SI << " (m), t0 = " << t0_SI << " (s)\n"
        << "With that we have:\n"
        << "cs2 = " << cs2 << " (lattice unit)\n"
        << "Ex_ext = " << Ex_ext << " (lattice unit), Ey_ext = " << Ey_ext << " (lattice unit)\n"
        << "T_e_init = " << T_e_init << " (lattice unit), T_i_init = " << T_i_init << " (lattice unit)\n"
        << "and equilibrium parameters:\n"
        << "tau_e = " << tau_e << ", tau_i = " << tau_i << ", tau_n = " << tau_n << "\n"
        << "tau_e_i = " << tau_e_i << ", tau_e_n = " << tau_e_n << ", tau_i_n = " << tau_i_n << "\n"
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
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t x = 0; x < NX; ++x) {
        for(size_t y = 0; y < NY; ++y){
            for (size_t i=0;i<Q;++i){
                const size_t idx_3 = INDEX(x, y, i);

                if (x<(3*NX/4) && x>(NX/4) && y<(3*NY/4) && y>(1*NY/4)){
                    f_e[idx_3] = w[i] * rho_e_init; // Equilibrium function for electrons
                    g_e[idx_3] = w[i] * T_e_init; // Thermal function for electrons
                    f_i[idx_3] = w[i] * rho_i_init; // Equilibrium function for ions
                    g_i[idx_3] = w[i] * T_i_init; // Thermal function for ions
                }
                f_n[idx_3] = w[i] * rho_n_init;
                g_n[idx_3] = w[i] * T_n_init;
            }
        }
    }
}
//──────────────────────────────────────────────────────────────────────────────
//  Compute D2Q9 equilibrium for a given (ρ, ux, uy, T) and sound‐speed cs2.
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::ComputeEquilibrium() {//It's the same for all the species, maybe can be used as a function
    // Compute the equilibrium distribution function f_eq
    #pragma omp parallel for collapse(2) schedule(static)
        for (size_t x = 0; x < NX; ++x) {
            for (size_t y = 0; y < NY; ++y) {
                const size_t idx = INDEX(x, y); // Get 1D index for 2D point (x, y)
                const double u2_e = ux_e[idx] * ux_e[idx] + uy_e[idx] * uy_e[idx]; // Square of the speed magnitude
                const double u2_i = ux_i[idx] * ux_i[idx] + uy_i[idx] * uy_i[idx]; 
                const double u2_n = ux_n[idx] * ux_n[idx] + uy_n[idx] * uy_n[idx]; 
                const double u2_e_i = ux_e_i[idx] * ux_e_i[idx] + uy_e_i[idx] * uy_e_i[idx]; // Square of the speed magnitude for electron-ion interaction
                const double u2_e_n = ux_e_n[idx] * ux_e_n[idx] + uy_e_n[idx] * uy_e_n[idx];
                const double u2_i_n = ux_i_n[idx] * ux_i_n[idx] + uy_i_n[idx] * uy_i_n[idx];
                const double den_e = rho_e[idx]; // Electron density
                const double den_i = rho_i[idx]; // Ion density
                const double den_n = rho_n[idx]; // Neutrals density
                
                for (size_t i = 0; i < Q; ++i) {
                    const size_t idx_3=INDEX(x, y, i);
                    const double cu_e = cx[i]*ux_e[idx] +cy[i]*uy_e[idx]; // Dot product (c_i · u)
                    const double cu_i = cx[i]*ux_i[idx] +cy[i]*uy_i[idx];
                    const double cu_n = cx[i]*ux_n[idx] +cy[i]*uy_n[idx];
                    const double cu_e_i = cx[i]*ux_e_i[idx] +cy[i]*uy_e_i[idx]; // Dot product (c_i · u) for electron-ion interaction
                    const double cu_e_n = cx[i]*ux_e_n[idx] +cy[i]*uy_e_n[idx];
                    const double cu_i_n = cx[i]*ux_i_n[idx] +cy[i]*uy_i_n[idx];

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
                    f_eq_n[idx_3]= w[i]*den_n*(
                        1.0 +
                        (cu_n / cs2) +
                        (cu_n * cu_n) / (2.0 * cs2 * cs2) -
                        u2_n / (2.0 * cs2)
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
                    f_eq_e_n[idx_3]= w[i]*den_e*(
                        1.0 +
                        (cu_e_n / cs2) +
                        (cu_e_n * cu_e_n) / (2.0 * cs2 * cs2) -
                        u2_e_n / (2.0 * cs2)
                    );
                    f_eq_n_e[idx_3]= w[i]*den_n*(
                        1.0 +
                        (cu_e_n / cs2) +
                        (cu_e_n * cu_e_n) / (2.0 * cs2 * cs2) -
                        u2_e_n / (2.0 * cs2)
                    );
                    f_eq_i_n[idx_3]= w[i]*den_i*(
                        1.0 +
                        (cu_i_n / cs2) +
                        (cu_i_n * cu_i_n) / (2.0 * cs2 * cs2) -
                        u2_i_n / (2.0 * cs2)
                    );
                    f_eq_n_i[idx_3]= w[i]*den_n*(
                        1.0 +
                        (cu_i_n / cs2) +
                        (cu_i_n * cu_i_n) / (2.0 * cs2 * cs2) -
                        u2_i_n / (2.0 * cs2)
                    );

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
                    g_eq_n[idx_3]=w[i]*T_n[idx]*(
                        1.0 +
                        (cu_n / cs2) +
                        (cu_n * cu_n ) / (2.0 * cs2 *cs2) -
                        u2_n / (2.0 * cs2)
                    );
                    g_eq_e_i[idx_3]= w[i]*T_e[idx]*(
                        1.0 +
                        (cu_e_i / cs2) +
                        (cu_e_i * cu_e_i) / (2.0 * cs2 * cs2) -
                        u2_e_i / (2.0 * cs2)
                    );
                    g_eq_i_e[idx_3]= w[i]*T_i[idx]*(
                        1.0 +
                        (cu_e_i / cs2) +
                        (cu_e_i * cu_e_i) / (2.0 * cs2 * cs2) -
                        u2_e_i / (2.0 * cs2)
                    );
                    g_eq_e_n[idx_3]= w[i]*T_e[idx]*(
                        1.0 +
                        (cu_e_n / cs2) +
                        (cu_e_n * cu_e_n) / (2.0 * cs2 * cs2) -
                        u2_e_n / (2.0 * cs2)
                    );
                    g_eq_n_e[idx_3]= w[i]*T_n[idx]*(
                        1.0 +
                        (cu_e_n / cs2) +
                        (cu_e_n * cu_e_n) / (2.0 * cs2 * cs2) -
                        u2_e_n / (2.0 * cs2)
                    );
                    g_eq_i_n[idx_3]= w[i]*T_i[idx]*(
                        1.0 +
                        (cu_i_n / cs2) +
                        (cu_i_n * cu_i_n) / (2.0 * cs2 * cs2) -
                        u2_i_n / (2.0 * cs2)
                    );
                    g_eq_n_i[idx_3]= w[i]*T_n[idx]*(
                        1.0 +
                        (cu_i_n / cs2) +
                        (cu_i_n * cu_i_n) / (2.0 * cs2 * cs2) -
                        u2_i_n / (2.0 * cs2)
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

    double rho_loc_n = 0.0;
    double ux_loc_n = 0.0;
    double uy_loc_n = 0.0;
    double T_loc_n = 0.0;

    #pragma omp parallel for collapse(2) private(rho_loc_e, ux_loc_e, uy_loc_e, T_loc_e, rho_loc_i, ux_loc_i, uy_loc_i, T_loc_i, rho_loc_n, ux_loc_n, uy_loc_n, T_loc_n)
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

                rho_loc_n = 0.0;
                ux_loc_n = 0.0;
                uy_loc_n = 0.0;
                T_loc_n = 0.0;

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

                    const double fi_n=f_n[idx_3];
                    rho_loc_n += fi_n;
                    ux_loc_n += fi_n * cx[i];
                    uy_loc_n += fi_n * cy[i];
                    T_loc_n += g_n[idx_3];
                    
                }
                if (rho_loc_e<1e-10){//in order to avoid instabilities
                    rho_e[idx] = 0.0;
                    ux_e[idx] = 0.0;
                    uy_e[idx] = 0.0;
                    T_e[idx] = 0.0;
                } else{
                    rho_e[idx] = rho_loc_e;
                    if(ux_loc_e==rho_loc_e || ux_loc_e==-rho_loc_e)
                        ux_e[idx]=0.01*cs2;
                    else
                        ux_e[idx] = ux_loc_e / rho_loc_e;
                    if(uy_loc_e==rho_loc_e || uy_loc_e==-rho_loc_e)
                        uy_e[idx]=0.01*cs2;
                    else
                        uy_e[idx] = uy_loc_e / rho_loc_e;
                    
                    ux_e[idx] += 0.5 * q_e * Ex[idx] / m_e;
                    uy_e[idx] += 0.5 * q_e * Ey[idx] / m_e;
                    T_e[idx] = T_loc_e;
                } 
                if (rho_loc_i<1e-10){
                    rho_i[idx] = 0.0;
                    ux_i[idx] = 0.0;
                    uy_i[idx] = 0.0;
                    T_i[idx] = 0.0;
                }else {
                    rho_i[idx] = rho_loc_i;
                    if(ux_loc_i==rho_loc_i || ux_loc_i==-rho_loc_i)
                        ux_i[idx]=0.00001*cs2;
                    else
                        ux_i[idx] = ux_loc_i / rho_loc_i;
                    if(uy_loc_i==rho_loc_i || uy_loc_i==-rho_loc_i)
                        uy_i[idx]=0.00001*cs2;
                    else
                        uy_i[idx] = uy_loc_i / rho_loc_i;

                    ux_i[idx] += 0.5 * q_i * Ex[idx] / m_i;
                    uy_i[idx] += 0.5 * q_i * Ey[idx] / m_i;
                    T_i[idx] = T_loc_i;
                }
                if (rho_loc_n<1e-10){
                    rho_n[idx] = 0.0;
                    ux_n[idx] = 0.0;
                    uy_n[idx] = 0.0;
                    T_n[idx] = 0.0;
                }
                else {
                    rho_n[idx] = rho_loc_n;
                    ux_n[idx] = ux_loc_n / rho_loc_n;
                    uy_n[idx] = uy_loc_n / rho_loc_n;
                    //No force term for neutrals, they are not charged
                    T_n[idx] = T_loc_n;
                }
                if (rho_loc_e<1e-10 && rho_loc_i<1e-10){
                    ux_e_i[idx] = 0.0;
                    uy_e_i[idx] = 0.0;
                }
                else {
                    ux_e_i[idx] = (rho_loc_e * ux_e[idx] + rho_loc_i * ux_i[idx]) / (rho_loc_e + rho_loc_i);
                    uy_e_i[idx] = (rho_loc_e * uy_e[idx] + rho_loc_i * uy_i[idx]) / (rho_loc_e + rho_loc_i);
                }
                if (rho_loc_e<1e-10 && rho_loc_n<1e-10){
                    ux_e_n[idx] = 0.0;
                    uy_e_n[idx] = 0.0;
                }
                else {
                    ux_e_n[idx] = (rho_loc_e * ux_e[idx] + rho_loc_n * ux_n[idx]) / (rho_loc_e + rho_loc_n);
                    uy_e_n[idx] = (rho_loc_e * uy_e[idx] + rho_loc_n * uy_n[idx]) / (rho_loc_e + rho_loc_n);
                }
                if (rho_loc_i<1e-10 && rho_loc_n<1e-10){
                    ux_i_n[idx] = 0.0;
                    uy_i_n[idx] = 0.0;
                }
                else {
                    ux_i_n[idx] = (rho_loc_i * ux_i[idx] + rho_loc_n * ux_n[idx]) / (rho_loc_i + rho_loc_n);
                    uy_i_n[idx] = (rho_loc_i * uy_i[idx] + rho_loc_n * uy_n[idx]) / (rho_loc_i + rho_loc_n);
                }
                
                // Lattice‐unit charge density (#/cell difference):
                rho_q[idx] = (q_i * rho_i[idx] / m_i + q_e * rho_e[idx] / m_e);
                if(rho_q[idx]<1e-15)    rho_q[idx]=0.0; //correct for machine error
            }
        }
}
//──────────────────────────────────────────────────────────────────────────────
//  Poisson dispatcher:
//  - 5-point stencil: \nabla^2 φ = -ρ_q,  φ_new = 1/4(φ_E + φ_W + φ_N + φ_S + RHS)
//  - SOR: φ = (1-ω)φ_old + ω φ_GS
//  - 9-point stencil: φ_new = [4*(orthogonal neighbors) + (diagonals) + 6*RHS] / 20
//  - FFT: use discrete sine transform in Fourier space
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::SolvePoisson() {
    // Dispatcher for Poisson solvers
    if (poisson_type == PoissonType::GAUSS_SEIDEL) {
        if (bc_type == BCType::PERIODIC) SolvePoisson_GS_Periodic();
        else SolvePoisson_GS();
    } else if (poisson_type == PoissonType::SOR) {
        if (bc_type == BCType::PERIODIC) SolvePoisson_SOR_Periodic();
        else SolvePoisson_SOR();
    } else if (poisson_type == PoissonType::FFT && bc_type == BCType::PERIODIC) {
        SolvePoisson_fft();
    } else if (poisson_type == PoissonType::NPS) {
        if (bc_type == BCType::PERIODIC) SolvePoisson_9point_Periodic();
        else SolvePoisson_9point();
    }
}

    
//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: Gauss–Seidel with Dirichlet φ=0 on boundary.
//    We solve  ∇² φ = − ρ_q_phys / ε₀,  in lattice units.
//    Our RHS in “lattice‐Land” is:  RHS_latt = − ρ_q_latt.
//    Then φ_new[i,j] = ¼ [ φ[i+1,j] + φ[i−1,j] + φ[i,j+1] + φ[i,j−1] − RHS_latt[i,j] ].
//
//  After convergence, we reconstruct E with centered differences:
//    E_x = −(φ[i+1,j] − φ[i−1,j]) / (2), etc.
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::SolvePoisson_GS() {
    // Parameters for the Gauss–Seidel solver
    const size_t maxIter = 5000;     // maximum number of iterations
    const double tol    = 1e-8;      // convergence tolerance

    // Red–Black Gauss–Seidel iteration for parallelism:
    // we split the grid into two interleaved sets ("red" and "black")
    // so each set can be updated in parallel without data races.
    for (size_t iter = 0; iter < maxIter; ++iter) {
        double maxErr = 0.0;

        // Update "red" points: (i+j) even
        #pragma omp parallel for collapse(2) reduction(max:maxErr)
        for (size_t j = 1; j < NY - 1; ++j) {
            for (size_t i = 1; i < NX - 1; ++i) {
                if (((i + j) & 1) == 0) {
                    const size_t idx = INDEX(i, j);
                    // sum of four neighbors
                    const double nb = phi[INDEX(i+1, j)]
                              + phi[INDEX(i-1, j)]
                              + phi[INDEX(i, j+1)]
                              + phi[INDEX(i, j-1)];
                    const double newPhi = 0.25 * (nb + rho_q[idx]);
                    const double err    = fabs(newPhi - phi[idx]);
                    phi[idx] = newPhi;
                    if (err > maxErr) maxErr = err;
                }
            }
        }

        // Update "black" points: (i+j) odd
        #pragma omp parallel for collapse(2) reduction(max:maxErr)
        for (size_t j = 1; j < NY - 1; ++j) {
            for (size_t i = 1; i < NX - 1; ++i) {
                if (((i + j) & 1) == 1) {
                    const size_t idx = INDEX(i, j);
                    const double nb = phi[INDEX(i+1, j)]
                              + phi[INDEX(i-1, j)]
                              + phi[INDEX(i, j+1)]
                              + phi[INDEX(i, j-1)];
                    const double newPhi = 0.25 * (nb + rho_q[idx]);
                    const double err    = fabs(newPhi - phi[idx]);
                    phi[idx] = newPhi;
                    if (err > maxErr) maxErr = err;
                }
            }
        }

        // Check for convergence
        if (maxErr < tol) {
            // early exit if solution has converged
            break;
        }
    }

    // Compute electric field E = -∇φ using central differences
    // Only interior points; boundaries will be set by Neumann BC below
    #pragma omp parallel for collapse(2)
    for (size_t j = 1; j < NY - 1; ++j) {
        for (size_t i = 1; i < NX - 1; ++i) {
            const size_t idx = INDEX(i, j);
            Ex[idx] = -0.5 * (phi[INDEX(i+1, j)] - phi[INDEX(i-1, j)]);
            Ey[idx] = -0.5 * (phi[INDEX(i, j+1)] - phi[INDEX(i, j-1)]);
        }
    }

    // Zero-Neumann boundary conditions (zero normal derivative):
    // copy the adjacent interior field value to the boundary cell.

    // Top and bottom boundaries
    #pragma omp parallel for
    for (size_t i = 0; i < NX; ++i) {
        Ex[INDEX(i, 0)]      = Ex[INDEX(i, 1)];
        Ey[INDEX(i, 0)]      = Ey[INDEX(i, 1)];
        Ex[INDEX(i, NY-1)]   = Ex[INDEX(i, NY-2)];
        Ey[INDEX(i, NY-1)]   = Ey[INDEX(i, NY-2)];
    }

    // Left and right boundaries
    #pragma omp parallel for
    for (size_t j = 0; j < NY; ++j) {
        Ex[INDEX(0, j)]      = Ex[INDEX(1, j)];
        Ey[INDEX(0, j)]      = Ey[INDEX(1, j)];
        Ex[INDEX(NX-1, j)]   = Ex[INDEX(NX-2, j)];
        Ey[INDEX(NX-1, j)]   = Ey[INDEX(NX-2, j)];
    }
}


//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: GS when BC are periodic for consistency.
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::SolvePoisson_GS_Periodic() {
    const size_t maxIter = 5000;       // max iterations
    const double tol    = 1e-8;        // convergence tolerance

    // Precompute strides for modulo arithmetic if you want, 
    // but here we compute them on the fly for clarity.
    for (size_t iter = 0; iter < maxIter; ++iter) {
        double maxErr = 0.0;

        // === RED sweep (i+j even) ===
        #pragma omp parallel for collapse(2) reduction(max:maxErr)
        for (size_t j = 0; j < NY; ++j) {
            for (size_t i = 0; i < NX; ++i) {
                if (((i + j) & 1) == 0) {
                    // periodic neighbors
                    const size_t ip = (i + 1) % NX;
                    const size_t im = (i + NX - 1) % NX;
                    const size_t jp = (j + 1) % NY;
                    const size_t jm = (j + NY - 1) % NY;

                    const size_t idx = INDEX(i, j);
                    const double sumNb =
                        phi[INDEX(ip, j)] +
                        phi[INDEX(im, j)] +
                        phi[INDEX(i, jp)] +
                        phi[INDEX(i, jm)];

                    const double newPhi = 0.25 * (sumNb + rho_q[idx]);
                    const double err    = fabs(newPhi - phi[idx]);
                    phi[idx]      = newPhi;

                    if (err > maxErr) maxErr = err;
                }
            }
        }

        // === BLACK sweep (i+j odd) ===
        #pragma omp parallel for collapse(2) reduction(max:maxErr)
        for (size_t j = 0; j < NY; ++j) {
            for (size_t i = 0; i < NX; ++i) {
                if (((i + j) & 1) == 1) {
                    // periodic neighbors
                    const size_t ip = (i + 1) % NX;
                    const size_t im = (i + NX - 1) % NX;
                    const size_t jp = (j + 1) % NY;
                    const size_t jm = (j + NY - 1) % NY;

                    const size_t idx = INDEX(i, j);
                    const double sumNb =
                        phi[INDEX(ip, j)] +
                        phi[INDEX(im, j)] +
                        phi[INDEX(i, jp)] +
                        phi[INDEX(i, jm)];

                    const double newPhi = 0.25 * (sumNb + rho_q[idx]);
                    const double err    = fabs(newPhi - phi[idx]);
                    phi[idx]      = newPhi;

                    if (err > maxErr) maxErr = err;
                }
            }
        }

        // Convergence check
        if (maxErr < tol) {
            break;
        }
    }

    // Compute periodic E field (can be done in one parallel pass)
    #pragma omp parallel for collapse(2)
    for (size_t j = 0; j < NY; ++j) {
        for (size_t i = 0; i < NX; ++i) {
            const size_t ip = (i + 1) % NX;
            const size_t im = (i + NX - 1) % NX;
            const size_t jp = (j + 1) % NY;
            const size_t jm = (j + NY - 1) % NY;
            const size_t idx = INDEX(i, j);

            Ex[idx] = -0.5 * (phi[INDEX(ip, j)] - phi[INDEX(im, j)]);
            Ey[idx] = -0.5 * (phi[INDEX(i, jp)] - phi[INDEX(i, jm)]);
        }
    }
}
//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: SOR (over‐relaxed Gauss–Seidel).  
//  Identical 5‐point stencil as GS, but φ_new = (1−ω) φ_old + ω φ_GS.
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::SolvePoisson_SOR() {
    // Parameters for SOR solver
    const size_t maxIter       = 5000;    // maximum iterations
    const double tol           = 1e-8;    // convergence tolerance
    const double omega         = omega_sor;   // over‐relaxation factor

    for (size_t iter = 0; iter < maxIter; ++iter) {
        double maxErr = 0.0;

        // === RED sweep (i+j even) ===
        #pragma omp parallel for collapse(2) reduction(max:maxErr)
        for (size_t j = 1; j < NY - 1; ++j) {
            for (size_t i = 1; i < NX - 1; ++i) {
                if (((i + j) & 1) == 0) {
                    const size_t idx = INDEX(i, j);
                    const double oldPhi = phi[idx];

                    // sum of neighbors (Gauss–Seidel stencil)
                    const double nb = phi[INDEX(i+1, j)]
                              + phi[INDEX(i-1, j)]
                              + phi[INDEX(i, j+1)]
                              + phi[INDEX(i, j-1)];

                    // standard Gauss–Seidel update
                    const double gsPhi = 0.25 * (nb + rho_q[idx]);

                    // SOR update: blend old and GS value
                    const double newPhi = (1.0 - omega) * oldPhi
                                  + omega       * gsPhi;

                    phi[idx] = newPhi;
                    const double err = fabs(newPhi - oldPhi);
                    if (err > maxErr) maxErr = err;
                }
            }
        }

        // === BLACK sweep (i+j odd) ===
        #pragma omp parallel for collapse(2) reduction(max:maxErr)
        for (size_t j = 1; j < NY - 1; ++j) {
            for (size_t i = 1; i < NX - 1; ++i) {
                if (((i + j) & 1) == 1) {
                    const size_t idx = INDEX(i, j);
                    const double oldPhi = phi[idx];

                    const double nb = phi[INDEX(i+1, j)]
                              + phi[INDEX(i-1, j)]
                              + phi[INDEX(i, j+1)]
                              + phi[INDEX(i, j-1)];

                    const double gsPhi = 0.25 * (nb + rho_q[idx]);
                    const double newPhi = (1.0 - omega) * oldPhi
                                  + omega       * gsPhi;

                    phi[idx] = newPhi;
                    const double err = fabs(newPhi - oldPhi);
                    if (err > maxErr) maxErr = err;
                }
            }
        }

        // check convergence
        if (maxErr < tol) {
            break;  // solution converged early
        }
    }

    // Compute interior E field (central differences)
    #pragma omp parallel for collapse(2)
    for (size_t j = 1; j < NY - 1; ++j) {
        for (size_t i = 1; i < NX - 1; ++i) {
            const size_t idx = INDEX(i, j);
            Ex[idx] = -0.5 * (phi[INDEX(i+1, j)] - phi[INDEX(i-1, j)]);
            Ey[idx] = -0.5 * (phi[INDEX(i, j+1)] - phi[INDEX(i, j-1)]);
        }
    }

    // Zero‑Neumann BC: copy adjacent interior value to boundary
    #pragma omp parallel for
    for (size_t i = 0; i < NX; ++i) {
        Ex[INDEX(i,   0)] = Ex[INDEX(i,   1)];
        Ey[INDEX(i,   0)] = Ey[INDEX(i,   1)];
        Ex[INDEX(i,NY-1)] = Ex[INDEX(i,NY-2)];
        Ey[INDEX(i,NY-1)] = Ey[INDEX(i,NY-2)];
    }
    #pragma omp parallel for
    for (size_t j = 0; j < NY; ++j) {
        Ex[INDEX(  0, j)] = Ex[INDEX(  1, j)];
        Ey[INDEX(  0, j)] = Ey[INDEX(  1, j)];
        Ex[INDEX(NX-1, j)] = Ex[INDEX(NX-2, j)];
        Ey[INDEX(NX-1, j)] = Ey[INDEX(NX-2, j)];
    }
}
//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: SOR when BC are periodic for consistency.
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::SolvePoisson_SOR_Periodic() {
    const size_t maxIter = 5000;           // maximum iterations
    const double tol    = 1e-8;            // convergence tolerance
    const double omega  = omega_sor;       // relaxation parameter

    for (size_t iter = 0; iter < maxIter; ++iter) {
        double maxErr = 0.0;

        // === RED sweep (i+j even) ===
        #pragma omp parallel for collapse(2) reduction(max:maxErr)
        for (size_t j = 0; j < NY; ++j) {
            for (size_t i = 0; i < NX; ++i) {
                if (((i + j) & 1) == 0) {
                    // periodic neighbor indices
                    const size_t ip = (i + 1) % NX;
                    const size_t im = (i + NX - 1) % NX;
                    const size_t jp = (j + 1) % NY;
                    const size_t jm = (j + NY - 1) % NY;

                    const size_t idx = INDEX(i, j);
                    const double oldPhi = phi[idx];

                    // Gauss–Seidel stencil sum
                    const double sumNb = phi[INDEX(ip, j)]
                                 + phi[INDEX(im, j)]
                                 + phi[INDEX(i, jp)]
                                 + phi[INDEX(i, jm)];

                    const double gsPhi   = 0.25 * (sumNb + rho_q[idx]);
                    const double newPhi  = (1.0 - omega) * oldPhi
                                   + omega       * gsPhi;
                    phi[idx]       = newPhi;

                    const double err = fabs(newPhi - oldPhi);
                    if (err > maxErr) maxErr = err;
                }
            }
        }

        // === BLACK sweep (i+j odd) ===
        #pragma omp parallel for collapse(2) reduction(max:maxErr)
        for (size_t j = 0; j < NY; ++j) {
            for (size_t i = 0; i < NX; ++i) {
                if (((i + j) & 1) == 1) {
                    const size_t ip = (i + 1) % NX;
                    const size_t im = (i + NX - 1) % NX;
                    const size_t jp = (j + 1) % NY;
                    const size_t jm = (j + NY - 1) % NY;

                    const size_t idx = INDEX(i, j);
                    const double oldPhi = phi[idx];

                    const double sumNb = phi[INDEX(ip, j)]
                                 + phi[INDEX(im, j)]
                                 + phi[INDEX(i, jp)]
                                 + phi[INDEX(i, jm)];

                    const double gsPhi   = 0.25 * (sumNb + rho_q[idx]);
                    const double newPhi  = (1.0 - omega) * oldPhi
                                   + omega       * gsPhi;
                    phi[idx]       = newPhi;

                    const double err = fabs(newPhi - oldPhi);
                    if (err > maxErr) maxErr = err;
                }
            }
        }

        // convergence check
        if (maxErr < tol) {
            break;
        }
    }

    // Compute periodic electric field E = -∇φ
    #pragma omp parallel for collapse(2)
    for (size_t j = 0; j < NY; ++j) {
        for (size_t i = 0; i < NX; ++i) {
            const size_t ip = (i + 1) % NX;
            const size_t im = (i + NX - 1) % NX;
            const size_t jp = (j + 1) % NY;
            const size_t jm = (j + NY - 1) % NY;

            const size_t idx = INDEX(i, j);
            Ex[idx] = -0.5 * (phi[INDEX(ip, j)] - phi[INDEX(im, j)]);
            Ey[idx] = -0.5 * (phi[INDEX(i, jp)] - phi[INDEX(i, jm)]);
        }
    }
}




//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: FFT (with periodic BCs).  
// Solves ∇²φ = –ρ_q by:
//   1) Forward real-to-complex FFT of rho_q → rho_hat(k)
//   2) Compute φ̂ (k) = –ρ̂ (k) / [4 sin²(π kx/NX) + 4 sin²(π ky/NY)]
//      (zeroing the k=0 mode to enforce zero-mean potential)
//   3) Inverse complex-to-real FFT of φ̂ → φ(x)
//   4) Normalize by NX*NY and reconstruct E = –∇φ via central differences
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::SolvePoisson_fft() {
    // Assumption: rho_q[i] is charge density in lattice units.
    // The net charge should be approximately zero.
    // If not, the k=0 component will be removed → potential average = 0.

    //index conversion: Since FFTW takes only int numbers we have to convert the grid to that values
    const int NXf = static_cast<int>(NX);
    const int NYf = static_cast<int>(NY);
    const int NYf_half = NYf / 2 + 1;//only NYf/2 +1 are unique coefficients
    const int real_size = NXf * NYf;//size of real space
    const int complex_size = NXf * NYf_half;//size of complex space

    // Allocate FFTW arrays
    // In order to solve a probelam in the Fourier space we need to convert things in the frequency domani, then solve and the convert back 
    double *in = (double*) fftw_malloc(sizeof(double) * real_size);                             //rho array in the real space
    fftw_complex *rho_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * complex_size);   //rho array in the complex space
    fftw_complex *phi_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * complex_size);   //phi array in the complex space
    double *out = (double*) fftw_malloc(sizeof(double) * real_size);                            //phi array in the real space

    // Copy input charge density into FFT input array
    #pragma omp parallel for
    for (int idx = 0; idx < real_size; ++idx) {
        in[idx] = rho_q[idx];
    }

    // Create FFTW plans (note: FFTW_ESTIMATE is thread-safe)
    fftw_plan plan_r2c = fftw_plan_dft_r2c_2d(NXf, NYf, in, rho_hat, FFTW_ESTIMATE);  //plane for the transformation from rho real to rho complex
    fftw_plan plan_c2r = fftw_plan_dft_c2r_2d(NXf, NYf, phi_hat, out, FFTW_ESTIMATE); //plane for the transformation from phi complex to phi real 
    
    // Forward FFT: rho_q -> rho_hat
    fftw_execute(plan_r2c);

    // Solve Poisson's equation in Fourier space:
    // ∇²φ = -rho → φ_hat = rho_hat / (kx² + ky²)
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NXf; ++i) {
        for (int j = 0; j < NYf_half; ++j) {
            const int kx = (i <= NXf / 2) ? i : i - NXf; //this ensures that kx index are centered around 0
            const int ky = j;
            const double sinx = std::sin(M_PI * kx / NXf);
            const double siny = std::sin(M_PI * ky / NYf);
            const double denom = 4.0 * (sinx * sinx + siny * siny);

            const size_t index = static_cast<size_t>(i) * NYf_half + j;

            if (denom > 1e-15) {
                phi_hat[index][0] = rho_hat[index][0] / denom;
                phi_hat[index][1] = rho_hat[index][1] / denom;
            } else {
                // Set zero mode to zero to enforce zero average potential (Gauge condition)
                // This is done for the zeroth mode φ_hat(0)
                phi_hat[index][0] = 0.0;
                phi_hat[index][1] = 0.0;
            }
        }
    }

    // Inverse FFT: phi_hat -> out
    fftw_execute(plan_c2r);// This is not yet normalized so we need to normalize it

    // Normalize result (FFTW does not normalize the inverse transform)
    const double norm = 1.0 / real_size; //Implicit conversion no need for casting
    #pragma omp parallel for
    for (int idx = 0; idx < real_size; ++idx) {
        phi[idx] = out[idx] * norm;
    }

    // Clean up FFTW allocations and plans
    fftw_destroy_plan(plan_r2c);
    fftw_destroy_plan(plan_c2r);
    fftw_free(in);
    fftw_free(out);
    fftw_free(rho_hat);
    fftw_free(phi_hat);

    // Compute electric field from potential using central differences
    // Periodic boundaries are handled with modulo indexing
    #pragma omp parallel for collapse(2)
    for (size_t j = 0; j < NY; ++j) {
        for (size_t i = 0; i < NX; ++i) {
            const size_t im1 = (i + NX - 1) % NX;
            const size_t ip1 = (i + 1) % NX;
            const size_t jm1 = (j + NY - 1) % NY;
            const size_t jp1 = (j + 1) % NY;
            const size_t idx = INDEX(i, j);

            Ex[idx] = -0.5 * (phi[INDEX(ip1, j)] - phi[INDEX(im1, j)]);
            Ey[idx] = -0.5 * (phi[INDEX(i, jp1)] - phi[INDEX(i, jm1)]);
        }
    }
}


//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: 9-point stencil with Dirichlet φ=0 on boundary.
//    We solve  ∇²_latt φ_latt = − ρ_q_latt.
//    Then φ_new = (4*neighbors + diagonals + 6*RHS) / 20.
//
//  After convergence, we reconstruct E with centered differences:
//    E_x = −(φ[i+1,j] − φ[i−1,j]) / (2), etc.
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::SolvePoisson_9point() {
    const size_t maxIter = 5000;    // maximum iterations
    const double tol     = 1e-8;    // convergence tolerance

    // Iterate until convergence or maxIter
    for (size_t iter = 0; iter < maxIter; ++iter) {
        double maxErr = 0.0;

        // We need 4 colors so that no two stencil neighbors share the same color.
        // color = 2*(i%2) + (j%2) ∈ {0,1,2,3}
        for (size_t sweep = 0; sweep < 4; ++sweep) {
            double maxErrSweep = 0.0;

            #pragma omp parallel for collapse(2) reduction(max:maxErrSweep)
            for (size_t j = 1; j < NY - 1; ++j) {
                for (size_t i = 1; i < NX - 1; ++i) {
                    // determine checkerboard color
                    if ((2 * (i & 1) + (j & 1)) != sweep) 
                        continue;

                    const size_t idx       = INDEX(i, j);
                    const size_t ip1_j     = INDEX(i+1, j);
                    const size_t im1_j     = INDEX(i-1, j);
                    const size_t i_jp1     = INDEX(i, j+1);
                    const size_t i_jm1     = INDEX(i, j-1);
                    const size_t ip1_jp1   = INDEX(i+1, j+1);
                    const size_t im1_jp1   = INDEX(i-1, j+1);
                    const size_t ip1_jm1   = INDEX(i+1, j-1);
                    const size_t im1_jm1   = INDEX(i-1, j-1);

                    // orthogonal neighbors
                    const double sumOrtho = phi[ip1_j] + phi[im1_j]
                                    + phi[i_jp1] + phi[i_jm1];
                    // diagonal neighbors
                    const double sumDiag  = phi[ip1_jp1] + phi[im1_jp1]
                                    + phi[ip1_jm1] + phi[im1_jm1];

                    // 9‑point update
                    const double newPhi = (4.0*sumOrtho + sumDiag + 6.0*rho_q[idx]) / 20.0;
                    const double err    = fabs(newPhi - phi[idx]);
                    phi[idx]      = newPhi;

                    if (err > maxErrSweep) maxErrSweep = err;
                }
            }

            // combine sweep error into global error
            if (maxErrSweep > maxErr) maxErr = maxErrSweep;
        }

        // check convergence
        if (maxErr < tol) {
            break;
        }
    }

    // Compute electric field (central differences) for interior nodes
    #pragma omp parallel for collapse(2)
    for (size_t j = 1; j < NY - 1; ++j) {
        for (size_t i = 1; i < NX - 1; ++i) {
            const size_t idx = INDEX(i, j);
            Ex[idx] = -0.5 * (phi[INDEX(i+1, j)] - phi[INDEX(i-1, j)]);
            Ey[idx] = -0.5 * (phi[INDEX(i, j+1)] - phi[INDEX(i, j-1)]);
        }
    }

    // Zero‑Neumann BC: copy nearest interior field to boundaries
    #pragma omp parallel for
    for (size_t i = 0; i < NX; ++i) {
        Ex[INDEX(i,   0)] = Ex[INDEX(i,   1)];
        Ey[INDEX(i,   0)] = Ey[INDEX(i,   1)];
        Ex[INDEX(i,NY-1)] = Ex[INDEX(i,NY-2)];
        Ey[INDEX(i,NY-1)] = Ey[INDEX(i,NY-2)];
    }
    #pragma omp parallel for
    for (size_t j = 0; j < NY; ++j) {
        Ex[INDEX(  0, j)] = Ex[INDEX(  1, j)];
        Ey[INDEX(  0, j)] = Ey[INDEX(  1, j)];
        Ex[INDEX(NX-1, j)] = Ex[INDEX(NX-2, j)];
        Ey[INDEX(NX-1, j)] = Ey[INDEX(NX-2, j)];
    }
}
//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: 9-point stencil when BC are periodic for consistency.
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::SolvePoisson_9point_Periodic() {
    const size_t maxIter = 5000;    // maximum iterations
    const double tol     = 1e-8;    // convergence tolerance

    // Main GS iteration with 4‑color ordering
    for (size_t iter = 0; iter < maxIter; ++iter) {
        double maxErr = 0.0;

        // Perform one sweep for each of the 4 colors
        for (size_t sweep = 0; sweep < 4; ++sweep) {
            double maxErrSweep = 0.0;

            #pragma omp parallel for collapse(2) reduction(max:maxErrSweep) schedule(static)
            for (size_t j = 0; j < NY; ++j) {
                for (size_t i = 0; i < NX; ++i) {
                    // determine 4‑color index: 2*(i%2) + (j%2) in {0,1,2,3}
                    if ((2 * (i & 1) + (j & 1)) != sweep) 
                        continue;

                    // periodic neighbor indices
                    const size_t ip = (i + 1) % NX;
                    const size_t im = (i + NX - 1) % NX;
                    const size_t jp = (j + 1) % NY;
                    const size_t jm = (j + NY - 1) % NY;

                    // flattened indices
                    const size_t idx   = INDEX(i,  j);
                    const size_t idxE  = INDEX(ip, j);
                    const size_t idxW  = INDEX(im, j);
                    const size_t idxN  = INDEX(i,  jp);
                    const size_t idxS  = INDEX(i,  jm);
                    const size_t idxNE = INDEX(ip, jp);
                    const size_t idxNW = INDEX(im, jp);
                    const size_t idxSE = INDEX(ip, jm);
                    const size_t idxSW = INDEX(im, jm);

                    // orthogonal and diagonal neighbor sums
                    const double sumO = phi[idxE]  + phi[idxW]
                                + phi[idxN]  + phi[idxS];
                    const double sumD = phi[idxNE] + phi[idxNW]
                                + phi[idxSE] + phi[idxSW];

                    // 9‑point Gauss–Seidel update
                    const double newPhi = (4.0 * sumO + sumD + 6.0 * rho_q[idx]) * 0.05; // /20
                    const double err    = fabs(newPhi - phi[idx]);
                    phi[idx]      = newPhi;

                    if (err > maxErrSweep) maxErrSweep = err;
                }
            }

            // accumulate the worst error across all sweeps
            if (maxErrSweep > maxErr) maxErr = maxErrSweep;
        }

        // convergence check
        if (maxErr < tol) {
            break;
        }
    }

    // Reconstruct periodic E = –∇φ with central differences
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t j = 0; j < NY; ++j) {
        for (size_t i = 0; i < NX; ++i) {
            const size_t im  = (i + NX - 1) % NX;
            const size_t ip  = (i + 1) % NX;
            const size_t jm = (j + NY - 1) % NY;
            const size_t jp = (j + 1) % NY;
            const size_t idx = INDEX(i, j);

            Ex[idx] = -0.5 * (phi[INDEX(ip, j)] - phi[INDEX(im, j)]);
            Ey[idx] = -0.5 * (phi[INDEX(i, jp)] - phi[INDEX(i, jm)]);
        }
    }
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
                const double C_e = -(f_e[idx_3]-f_eq_e[idx_3]) / tau_e -(f_e[idx_3]-f_eq_e_i[idx_3]) / tau_e_i -(f_e[idx_3]-f_eq_e_n[idx_3]) / tau_e_n;
                const double C_i = -(f_i[idx_3]-f_eq_i[idx_3]) / tau_i -(f_i[idx_3]-f_eq_i_e[idx_3]) / tau_e_i -(f_i[idx_3]-f_eq_i_n[idx_3]) / tau_i_n;
                const double C_n = -(f_n[idx_3]-f_eq_n[idx_3]) / tau_n -(f_n[idx_3]-f_eq_n_e[idx_3]) / tau_e_n -(f_n[idx_3]-f_eq_n_i[idx_3]) / tau_i_n;

                // Update distribution functions with Guo forcing term
                f_temp_e[idx_3] = f_e[idx_3] + C_e + F_e;
                f_temp_i[idx_3] = f_i[idx_3] + C_i + F_i;
                f_temp_n[idx_3] = f_n[idx_3] + C_n;
            }
        }
    }
    // Swap temporary arrays with main arrays
    f_e.swap(f_temp_e);
    f_i.swap(f_temp_i);
    f_n.swap(f_temp_n);
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
                const size_t idx_2 = INDEX(x, y);

                const double term_ee=(2.0*rho_e[idx_2]*(1.0-1.0/tau_e)*(1-1/tau_e)-2.0*(1.0-1.0/tau_e)*rho_e[idx_2]-Q*f_eq_e[idx_3]/tau_e)/(2.0*(2.0*(1.0-1.0/tau_e)+Q*f_eq_e[idx_3]/tau_e));
                const double term_ei=(2.0*rho_e[idx_2]*(1.0-1.0/tau_e_i)*(1-1/tau_e_i)-2.0*(1.0-1.0/tau_e_i)*rho_e[idx_2]-Q*f_eq_e_i[idx_3]/tau_e_i)/(2.0*(2.0*(1.0-1.0/tau_e_i)+Q*f_eq_e_i[idx_3]/tau_e_i));
                const double term_en=(2.0*rho_e[idx_2]*(1.0-1.0/tau_e_n)*(1-1/tau_e_n)-2.0*(1.0-1.0/tau_e_n)*rho_e[idx_2]-Q*f_eq_e_n[idx_3]/tau_e_n)/(2.0*(2.0*(1.0-1.0/tau_e_n)+Q*f_eq_e_n[idx_3]/tau_e_n));

                const double term_ii=(2.0*rho_i[idx_2]*(1.0-1.0/tau_i)*(1-1/tau_i)-2.0*(1.0-1.0/tau_i)*rho_i[idx_2]-Q*f_eq_i[idx_3]/tau_i)/(2.0*(2.0*(1.0-1.0/tau_i)+Q*f_eq_i[idx_3]/tau_i));
                const double term_ie=(2.0*rho_i[idx_2]*(1.0-1.0/tau_e_i)*(1-1/tau_e_i)-2.0*(1.0-1.0/tau_e_i)*rho_i[idx_2]-Q*f_eq_i_e[idx_3]/tau_e_i)/(2.0*(2.0*(1.0-1.0/tau_e_i)+Q*f_eq_i_e[idx_3]/tau_e_i));
                const double term_in=(2.0*rho_i[idx_2]*(1.0-1.0/tau_i_n)*(1-1/tau_i_n)-2.0*(1.0-1.0/tau_i_n)*rho_i[idx_2]-Q*f_eq_i_n[idx_3]/tau_i_n)/(2.0*(2.0*(1.0-1.0/tau_i_n)+Q*f_eq_i_n[idx_3]/tau_i_n));

                const double term_nn=(2.0*rho_n[idx_2]*(1.0-1.0/tau_n)*(1-1/tau_n)-2.0*(1.0-1.0/tau_n)*rho_n[idx_2]-Q*f_eq_n[idx_3]/tau_n)/(2.0*(2.0*(1.0-1.0/tau_n)+Q*f_eq_n[idx_3]/tau_n));
                const double term_ne=(2.0*rho_n[idx_2]*(1.0-1.0/tau_e_n)*(1-1/tau_e_n)-2.0*(1.0-1.0/tau_e_n)*rho_n[idx_2]-Q*f_eq_n_e[idx_3]/tau_e_n)/(2.0*(2.0*(1.0-1.0/tau_e_n)+Q*f_eq_n_e[idx_3]/tau_e_n));
                const double term_ni=(2.0*rho_n[idx_2]*(1.0-1.0/tau_i_n)*(1-1/tau_i_n)-2.0*(1.0-1.0/tau_i_n)*rho_n[idx_2]-Q*f_eq_n_i[idx_3]/tau_i_n)/(2.0*(2.0*(1.0-1.0/tau_i_n)+Q*f_eq_n_i[idx_3]/tau_i_n));

                const double DeltaE_e= rho_e[idx_2]*(term_ee+ term_ei+term_en)*(ux_e[idx_2]*ux_e[idx_2]+uy_e[idx_2]*uy_e[idx_2]);
                const double DeltaE_i= rho_i[idx_2]*(term_ii+ term_ie+term_in)*(ux_i[idx_2]*ux_i[idx_2]+uy_i[idx_2]*uy_i[idx_2]);
                const double DeltaE_n= rho_n[idx_2]*(term_nn+ term_ne+term_ni)*(ux_n[idx_2]*ux_n[idx_2]+uy_n[idx_2]*uy_n[idx_2]);

                const double DeltaT_e= - DeltaE_e/Kb;
                const double DeltaT_i= - DeltaE_i/Kb;
                const double DeltaT_n= - DeltaE_n/Kb;

                // Compute complete collisions terms
                const double C_Te = -(g_e[idx_3]-g_eq_e[idx_3]) / tau_e -(g_e[idx_3]-g_eq_e_i[idx_3]) / tau_e_i -(g_e[idx_3]-g_eq_e_n[idx_3]) / tau_e_n;
                const double C_Ti = -(g_i[idx_3]-g_eq_i[idx_3]) / tau_i -(g_i[idx_3]-g_eq_i_e[idx_3]) / tau_e_i -(g_i[idx_3]-g_eq_i_n[idx_3]) / tau_i_n;
                const double C_Tn = -(g_n[idx_3]-g_eq_n[idx_3]) / tau_n -(g_n[idx_3]-g_eq_n_e[idx_3]) / tau_e_n -(g_n[idx_3]-g_eq_n_i[idx_3]) / tau_i_n;

                // Update distribution functions with Guo forcing term
                g_temp_e[idx_3] = g_e[idx_3]+ C_Te + DeltaT_e;
                g_temp_i[idx_3] = g_i[idx_3]+ C_Ti + DeltaT_i;
                g_temp_n[idx_3] = g_n[idx_3]+ C_Tn + DeltaT_n;
            }
        }
    }
    // Swap temporary arrays with main arrays
    g_e.swap(g_temp_e);
    g_i.swap(g_temp_i);
    g_n.swap(g_temp_n);
}
//──────────────────────────────────────────────────────────────────────────────
//  Streaming dispatcher:
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::Streaming() {
    if (bc_type == BCType::PERIODIC){Streaming_Periodic();    ThermalStreaming_Periodic();}
    else /* BCType::BOUNCE_BACK */  {Streaming_BounceBack();  ThermalStreaming_BounceBack();}
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
                f_temp_e[INDEX(x_str, y_str, i)] = f_e[INDEX(x, y, i)];
                f_temp_n[INDEX(x_str, y_str, i)] = f_n[INDEX(x, y, i)];
            }
        }
    }
    f_e.swap(f_temp_e);
    f_i.swap(f_temp_i);
    f_n.swap(f_temp_n);
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
                        f_temp_n[INDEX(x_str, y_str, i)] = f_n[INDEX(x, y, i)];
                    }
                    else if(x_str >= 0 && x_str <static_cast<int>(NX)){
                        //We are outside so bounceback
                        f_temp_e[INDEX(x_str,y,opp[i])]=f_e[INDEX(x,y,i)];
                        f_temp_i[INDEX(x_str,y,opp[i])]=f_i[INDEX(x,y,i)];
                        f_temp_n[INDEX(x_str,y,opp[i])]=f_n[INDEX(x,y,i)];
                    }
                    else if(y_str >= 0 && y_str <static_cast<int>(NY)){
                        //We are outside so bounceback
                        f_temp_e[INDEX(x,y_str,opp[i])]=f_e[INDEX(x,y,i)];
                        f_temp_i[INDEX(x,y_str,opp[i])]=f_i[INDEX(x,y,i)];
                        f_temp_n[INDEX(x,y_str,opp[i])]=f_n[INDEX(x,y,i)];
                    }
                    else{
                        //We are outside so bounceback
                        f_temp_e[INDEX(x,y,opp[i])]=f_e[INDEX(x,y,i)];
                        f_temp_i[INDEX(x,y,opp[i])]=f_i[INDEX(x,y,i)];
                        f_temp_n[INDEX(x,y,opp[i])]=f_n[INDEX(x,y,i)];
                    }
                }
            }
        }
    f_e.swap(f_temp_e);
    f_i.swap(f_temp_i);
    f_n.swap(f_temp_n);
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
void LBmethod::ThermalStreaming_Periodic() {
    
    //#pragma omp parallel for collapse(3) schedule(static)
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
            for (size_t i = 0; i < Q; ++i) {
                // Streaming coordinates (periodic wrapping)
                size_t x_str = (x + NX + cx[i]) % NX;
                size_t y_str = (y + NY + cy[i]) % NY;

                g_temp_i[INDEX(x_str, y_str, i)]  = g_i[INDEX(x, y, i)];
                g_temp_e[INDEX(x_str, y_str, i)]  = g_e[INDEX(x, y, i)];
                g_temp_n[INDEX(x_str, y_str, i)]  = g_n[INDEX(x, y, i)];
            }
        }
    }
    g_e.swap(g_temp_e);
    g_i.swap(g_temp_i);
    g_n.swap(g_temp_n);
}

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
                        g_temp_n[INDEX(x_str, y_str, i)] = g_n[INDEX(x, y, i)];
                    }
                    else if(x_str >= 0 && x_str <static_cast<int>(NX)){
                        //We are outside so bounceback
                        g_temp_e[INDEX(x_str,y,opp[i])]=g_e[INDEX(x,y,i)];
                        g_temp_i[INDEX(x_str,y,opp[i])]=g_i[INDEX(x,y,i)];
                        g_temp_n[INDEX(x_str,y,opp[i])]=g_n[INDEX(x,y,i)];
                    }
                    else if(y_str >= 0 && y_str <static_cast<int>(NY)){
                        //We are outside so bounceback
                        g_temp_e[INDEX(x,y_str,opp[i])]=g_e[INDEX(x,y,i)];
                        g_temp_i[INDEX(x,y_str,opp[i])]=g_i[INDEX(x,y,i)];
                        g_temp_n[INDEX(x,y_str,opp[i])]=g_n[INDEX(x,y,i)];
                    }
                    else{
                        //We are outside so bounceback
                        g_temp_e[INDEX(x,y,opp[i])]=g_e[INDEX(x,y,i)];
                        g_temp_i[INDEX(x,y,opp[i])]=g_i[INDEX(x,y,i)];
                        g_temp_n[INDEX(x,y,opp[i])]=g_n[INDEX(x,y,i)];
                    }
                }
            }
        }
    g_e.swap(g_temp_e);
    g_i.swap(g_temp_i);
    g_n.swap(g_temp_n);
}
void LBmethod::Run_simulation() {
    
    // Set threads for this simulation
    omp_set_num_threads(n_cores);

    // Inizializza CSV time series
    InitTimeSeries();

    InitDebugDump("debug_dump.txt");

    // Pre‐compute the frame sizes for each video:
    const int border       = 10;
    const int label_height = 30;
    const int tile_w       = NX + 2 * border;                // panel width
    const int tile_h       = NY + 2 * border + label_height; // panel height
    const double fps       = 1.0; // frames per second for videos

    // --- Density‐video (2 panels side by side) ---
    {
        int legend_width = 40, text_area = 60;
        int panel_width  = legend_width + text_area;
        int frame_w = 3 * tile_w + 2 * panel_width + 4 * border;
        int frame_h = tile_h;
        video_writer_density.open(
            "video_density.mp4",
            cv::VideoWriter::fourcc('m','p','4','v'),
            fps,                        // fps
            cv::Size(frame_w, frame_h),
            true                        // isColor
        );
        if (!video_writer_density.isOpened()) {
            std::cerr << "Cannot open video_density.mp4 for writing\n";
            return;
        }
    }

    // --- Velocity‐video (2 rows × 3 columns = 6 panels) ---
    {
        int frame_w = 3 * (NX + 2 * border);                         // 3 tiles in larghezza
        int frame_h = 2 * (NY + 2 * border + label_height);          // 2 tiles in altezza

        video_writer_velocity.open(
            "video_velocity.mp4",
            cv::VideoWriter::fourcc('m','p','4','v'),
            fps,
            cv::Size(frame_w, frame_h),
            true
        );
        if (!video_writer_velocity.isOpened()) {
            std::cerr << "Cannot open video_velocity.mp4 for writing\n";
            return;
        }
    }



    // --- Temperature‐video (2 panels side by side) ---
    {
        int frame_w = 2 * tile_w;
        int frame_h = tile_h;
        video_writer_temperature.open(
            "video_temperature.mp4",
            cv::VideoWriter::fourcc('m','p','4','v'),
            fps,
            cv::Size(frame_w, frame_h),
            true
        );
        if (!video_writer_temperature.isOpened()) {
            std::cerr << "Cannot open video_temperature.mp4 for writing\n";
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
        RecordTimeSeriesStep(t);
        UpdateMacro(); // rho=sum(f), ux=sum(f*c_x)/rho, uy=sum(f*c_y)/rho
        if(NX<11) DumpGridStateReadable(t, "UpdateMacro");

        if (t%1==0) {
            auto max_rho_e = std::max_element(rho_e.begin(), rho_e.end());
            auto min_rho_e = std::min_element(rho_e.begin(), rho_e.end());
            auto max_rho_i = std::max_element(rho_i.begin(), rho_i.end());
            auto min_rho_i = std::min_element(rho_i.begin(), rho_i.end());
            auto max_rho_n = std::max_element(rho_n.begin(), rho_n.end());
            auto min_rho_n = std::min_element(rho_n.begin(), rho_n.end());
            auto max_uxe = std::max_element(ux_e.begin(), ux_e.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto min_uxe = std::min_element(ux_e.begin(), ux_e.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto max_uye = std::max_element(uy_e.begin(), uy_e.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto min_uye = std::min_element(uy_e.begin(), uy_e.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto max_uxi = std::max_element(ux_i.begin(), ux_i.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto min_uxi = std::min_element(ux_i.begin(), ux_i.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto max_uyi = std::max_element(uy_i.begin(), uy_i.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto min_uyi = std::min_element(uy_i.begin(), uy_i.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto max_uxn = std::max_element(ux_n.begin(), ux_n.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto min_uxn = std::min_element(ux_n.begin(), ux_n.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto max_uyn = std::max_element(uy_n.begin(), uy_n.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
            auto min_uyn = std::min_element(uy_n.begin(), uy_n.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
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
            auto max_Tn = std::max_element(T_n.begin(), T_n.end());
            auto min_Tn = std::min_element(T_n.begin(), T_n.end());
            double totmass = 0.0;
            double totkinenerg = 0.0;
            double totT =0.0;
            for(size_t x=0;x<NX;++x){
                for(size_t y=0;y<NY;++y){
                    const size_t idx=INDEX(x,y);
                    totmass+=rho_e[idx]+rho_i[idx]+rho_n[idx];
                    totkinenerg+=0.5*rho_e[idx]*(ux_e[idx]*ux_e[idx]+uy_e[idx]*uy_e[idx])
                                +0.5*rho_i[idx]*(ux_i[idx]*ux_i[idx]+uy_i[idx]*uy_i[idx])
                                +0.5*rho_n[idx]*(ux_n[idx]*ux_n[idx]+uy_n[idx]*uy_n[idx]);
                    totT += ( T_e[idx] + T_i[idx] + T_n[idx]);

                }
            }

            std::cout <<"Step:"<<t<<std::endl;
            std::cout <<"max rho_e= "<<*max_rho_e<<", min rho_e= "<<*min_rho_e<<std::endl;
            std::cout <<"max rho_i= "<<*max_rho_i<<", min rho_i= "<<*min_rho_i<<std::endl;
            std::cout <<"max rho_n= "<<*max_rho_n<<", min rho_n= "<<*min_rho_n<<std::endl;
            std::cout <<"max ux_e= "<<*max_uxe<<", min ux_e= "<<*min_uxe<<std::endl;
            std::cout <<"max ux_i= "<<*max_uxi<<", min ux_i= "<<*min_uxi<<std::endl;
            std::cout <<"max ux_n= "<<*max_uxn<<", min ux_n= "<<*min_uxn<<std::endl;
            std::cout <<"max uy_e= "<<*max_uye<<", min uy_e= "<<*min_uye<<std::endl;
            std::cout <<"max uy_i= "<<*max_uyi<<", min uy_i= "<<*min_uyi<<std::endl;
            std::cout <<"max uy_n= "<<*max_uyn<<", min uy_n= "<<*min_uyn<<std::endl;
            std::cout <<"max Ex= "<<*max_Ex<<", min Ex= "<<*min_Ex<<std::endl;
            std::cout <<"max Ey= "<<*max_Ey<<", min Ey= "<<*min_Ey<<std::endl;
            std::cout <<"max T_e= "<<*max_Te<<", min T_e= "<<*min_Te<<std::endl;
            std::cout <<"max T_i= "<<*max_Ti<<", min T_i= "<<*min_Ti<<std::endl;
            std::cout <<"max T_n= "<<*max_Tn<<", min T_n= "<<*min_Tn<<std::endl;
            std::cout <<"max rho_q (latt)= "<<*max_rho<<", rho_q (latt)= "<<*min_rho<<std::endl;
            std::cout <<"Parameters to check:"<<std::endl;
            std::cout <<"totmass = "<<totmass<<" , totkinenerg= "<<totkinenerg<<" , totT= "<<totT<<std::endl;
            std::cout <<std::endl;
        }
        
        ComputeEquilibrium();
        if(NX<11) DumpGridStateReadable(t, "ComputeEquilibrium");
        ThermalCollisions();
        Collisions(); // f(x,y,t+1)=f(x-cx,y-cy,t) + tau * (f_eq - f) + dt*F
        if(NX<11) DumpGridStateReadable(t, "Collisions");
        Streaming(); // f(x,y,t+1)=f(x-cx,y-cy,t)
        if(NX<11) DumpGridStateReadable(t, "Streaming");
        SolvePoisson();
        if(NX<11) DumpGridStateReadable(t, "SolvePoisson");
        if (t==0){
            for(size_t x=0;x<NX;++x){
                for(size_t y=0;y<NY;++y){
                    Ex[INDEX(x,y)]=0.0;
                    Ey[INDEX(x,y)]=0.0;
                }
            }
        }
        VisualizationDensity();
        VisualizationVelocity();
        VisualizationTemperature();
    }

    video_writer_density.release();
    video_writer_velocity.release();
    video_writer_temperature.release();

    FinalizeTimeSeriesPlots();

    CloseDebugDump();

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
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
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

    auto c_n_e = normalize_and_color(mat_n_e, DENSITY_E_MIN, DENSITY_E_MAX);
    auto c_rho = normalize_and_color(mat_rho_q, CHARGE_MIN, CHARGE_MAX);
    auto c_n_i = normalize_and_color(mat_n_i, DENSITY_I_MIN, DENSITY_I_MAX);

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
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
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
    for (size_t x = 0; x < NX; ++x) {
        for (size_t y = 0; y < NY; ++y) {
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

void LBmethod::InitTimeSeries() {
    // define the 9 sample‐points
    size_t cx = NX/2, cy = NY/2;
    size_t dx = NX/4, dy = NY/4;
    sample_points = {
      {cx,cy}, {cx+dx,cy}, {cx-dx,cy}, {cx,cy+dy}, {cx,cy-dy},
      {cx+dx,cy+dy},{cx+dx,cy-dy},{cx-dx,cy+dy},{cx-dx,cy-dy}
    };

    // clear histories
    ts.clear();
    auto zero_hist = [&](auto &H){
      H.clear(); H.reserve(1024);
    };
    zero_hist(hist_ux_e);    zero_hist(hist_uy_e);    zero_hist(hist_ue_mag);
    zero_hist(hist_ux_i);    zero_hist(hist_uy_i);    zero_hist(hist_ui_mag);
    zero_hist(hist_ux_n);    zero_hist(hist_uy_n);    zero_hist(hist_un_mag);
    zero_hist(hist_T_e);     zero_hist(hist_T_i);     zero_hist(hist_T_n);
    zero_hist(hist_rho_e);   zero_hist(hist_rho_i);   zero_hist(hist_rho_n);
    zero_hist(hist_rho_q);   zero_hist(hist_Ex);      zero_hist(hist_Ey);
    zero_hist(hist_E_mag);
}


void LBmethod::RecordTimeSeriesStep(size_t t) {
    ts.push_back(t);
    size_t P = sample_points.size();

    // allocate one row per quantity
    auto push_row = [&](auto &H, auto compute){
      H.emplace_back(); 
      auto &row = H.back();
      row.resize(P);
      for(size_t p=0; p<P; ++p){
        auto [i,j] = sample_points[p];
        size_t idx = INDEX(i,j);
        row[p] = compute(idx);
      }
    };

    // electrons
    push_row(hist_ux_e,  [&](size_t idx){ return ux_e[idx]; });
    push_row(hist_uy_e,  [&](size_t idx){ return uy_e[idx]; });
    push_row(hist_ue_mag, [&](size_t idx){
      double x=ux_e[idx], y=uy_e[idx]; 
      return std::sqrt(x*x+y*y);
    });

    // ions
    push_row(hist_ux_i,  [&](size_t idx){ return ux_i[idx]; });
    push_row(hist_uy_i,  [&](size_t idx){ return uy_i[idx]; });
    push_row(hist_ui_mag, [&](size_t idx){
      double x=ux_i[idx], y=uy_i[idx];
      return std::sqrt(x*x+y*y);
    });

    // neutrals
    push_row(hist_ux_n,  [&](size_t idx){ return ux_n[idx]; });
    push_row(hist_uy_n,  [&](size_t idx){ return uy_n[idx]; });
    push_row(hist_un_mag, [&](size_t idx){
      double x=ux_n[idx], y=uy_n[idx];
      return std::sqrt(x*x+y*y);
    });

    // temperatures
    push_row(hist_T_e, [&](size_t idx){ return T_e[idx]; });
    push_row(hist_T_i, [&](size_t idx){ return T_i[idx]; });
    push_row(hist_T_n, [&](size_t idx){ return T_n[idx]; });

    // densities
    push_row(hist_rho_e, [&](size_t idx){ return rho_e[idx]; });
    push_row(hist_rho_i, [&](size_t idx){ return rho_i[idx]; });
    push_row(hist_rho_n, [&](size_t idx){ return rho_n[idx]; });
    push_row(hist_rho_q, [&](size_t idx){ return rho_q[idx]; });

    // fields
    push_row(hist_Ex, [&](size_t idx){ return Ex[idx]; });
    push_row(hist_Ey, [&](size_t idx){ return Ey[idx]; });
    push_row(hist_E_mag, [&](size_t idx){
      double x=Ex[idx], y=Ey[idx];
      return std::sqrt(x*x+y*y);
    });
}

void LBmethod::FinalizeTimeSeriesPlots() {
    // helper to build legends "p0( i_j )", …
    std::vector<std::string> legs;
    for(size_t p=0; p<sample_points.size(); ++p){
      auto [i,j]=sample_points[p];
      std::ostringstream ss;
      ss<<"p"<<p<<"("<<i<<"_"<<j<<")";
      legs.push_back(ss.str());
    }

    // call PlotTimeSeriesDirect for each quantity
    PlotTimeSeriesDirect("plot_ux_e.png","ux_e", legs, hist_ux_e);
    PlotTimeSeriesDirect("plot_uy_e.png","uy_e", legs, hist_uy_e);
    PlotTimeSeriesDirect("plot_ue_mag.png","|u_e|",legs, hist_ue_mag);
    PlotTimeSeriesDirect("plot_ux_i.png","ux_i", legs, hist_ux_i);
    PlotTimeSeriesDirect("plot_uy_i.png","uy_i", legs, hist_uy_i);
    PlotTimeSeriesDirect("plot_ui_mag.png","|u_i|",legs, hist_ui_mag);
    PlotTimeSeriesDirect("plot_ux_n.png","ux_n", legs, hist_ux_n);
    PlotTimeSeriesDirect("plot_uy_n.png","uy_n", legs, hist_uy_n);
    PlotTimeSeriesDirect("plot_un_mag.png","|u_n|",legs, hist_un_mag);
    PlotTimeSeriesDirect("plot_T_e.png","T_e",  legs, hist_T_e);
    PlotTimeSeriesDirect("plot_T_i.png","T_i",  legs, hist_T_i);
    PlotTimeSeriesDirect("plot_T_n.png","T_n",  legs, hist_T_n);
    PlotTimeSeriesDirect("plot_rho_e.png","rho_e", legs, hist_rho_e);
    PlotTimeSeriesDirect("plot_rho_i.png","rho_i", legs, hist_rho_i);
    PlotTimeSeriesDirect("plot_rho_n.png","rho_n", legs, hist_rho_n);
    PlotTimeSeriesDirect("plot_rho_q.png","rho_q", legs, hist_rho_q);
    PlotTimeSeriesDirect("plot_Ex.png","Ex",  legs, hist_Ex);
    PlotTimeSeriesDirect("plot_Ey.png","Ey",  legs, hist_Ey);
    PlotTimeSeriesDirect("plot_E_mag.png","|E|",legs, hist_E_mag);
}

void LBmethod::PlotTimeSeriesDirect(
    const std::string &png_filename,
    const std::string &title,
    const std::vector<std::string> &legends,
    const std::vector<std::vector<double>> &data)
{
    size_t Ncols = legends.size();
    size_t npts  = ts.size();
    if (npts < 2 || Ncols == 0) return;

    // find t‐range and value‐range
    double t_min = ts.front(), t_max = ts.back();
    double vmin =  1e300, vmax = -1e300;
    for (size_t c=0; c<Ncols; ++c)
      for (double v : data[c]) {
        vmin = std::min(vmin,v);
        vmax = std::max(vmax,v);
      }
    if (vmin == vmax) { vmin -= 1; vmax += 1; }

    // image setup
    int W=800, H=600;
    int ml=80, mr=40, mt=60, mb=80;
    cv::Mat img(H,W,CV_8UC3,cv::Scalar(255,255,255));
    cv::Point origin(ml,H-mb), xend(W-mr,H-mb), yend(ml,mt);
    cv::line(img,origin,xend,cv::Scalar(0,0,0));
    cv::line(img,origin,yend,cv::Scalar(0,0,0));
    cv::putText(img,title,cv::Point(ml,mt/2),
                cv::FONT_HERSHEY_SIMPLEX,0.8,cv::Scalar(0,0,0),2);

    double pw = double(W-ml-mr), ph = double(H-mt-mb);
    double xs = pw/(t_max-t_min), ys = ph/(vmax-vmin);

    // ticks
    int nt=5;
    for(int k=0;k<=nt;++k){
      double t = t_min + (t_max-t_min)*k/nt;
      int x = int(ml + (t-t_min)*xs +0.5);
      cv::line(img,{x,H-mb},{x,H-mb+5},cv::Scalar(0,0,0));
      std::ostringstream ss; ss<<std::fixed<<std::setprecision(0)<<t;
      cv::Size tsz = cv::getTextSize(ss.str(),cv::FONT_HERSHEY_SIMPLEX,0.5,1,0);
      cv::putText(img,ss.str(),{x-tsz.width/2,H-mb+20},
                  cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,0));
    }
    for(int k=0;k<=nt;++k){
      double v = vmin + (vmax-vmin)*k/nt;
      int y = int(H-mb - (v-vmin)*ys +0.5);
      cv::line(img,{ml,y},{ml-5,y},cv::Scalar(0,0,0));
      std::ostringstream ss; ss<<std::fixed<<std::setprecision(2)<<v;
      cv::Size tsz = cv::getTextSize(ss.str(),cv::FONT_HERSHEY_SIMPLEX,0.5,1,0);
      cv::putText(img,ss.str(),{ml-10-tsz.width,y+tsz.height/2},
                  cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,0));
    }

    // plot curves
    std::vector<cv::Scalar> cols = {
      {255,0,0},{0,128,0},{0,0,255},{255,165,0},
      {128,0,128},{0,255,255},{255,0,255},{128,128,0},{0,128,128}
    };
    for(size_t c=0; c<Ncols; ++c){
      std::vector<cv::Point> pts; pts.reserve(npts);
      for(size_t k=0;k<npts;++k){
        int x = int(ml + (ts[k]-t_min)*xs + 0.5);
        int y = int(H-mb - (data[c][k]-vmin)*ys + 0.5);
        pts.emplace_back(x,y);
      }
      auto col = cols[c%cols.size()];
      for(size_t k=1;k<pts.size(); ++k)
        cv::line(img,pts[k-1],pts[k],col,1);
    }

    // legend
    int lx=W-mr-150, ly=mt+10, lh=20;
    for(size_t c=0;c<Ncols; ++c){
      cv::rectangle(img,
        cv::Point(lx,ly+c*lh),
        cv::Point(lx+15,ly+15+c*lh),
        cols[c%cols.size()],cv::FILLED);
      cv::putText(img,legends[c],
        cv::Point(lx+20, ly+12+c*lh),
        cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,0),1);
    }

    // axis labels
    cv::putText(img,"time",
                cv::Point((ml+W-mr)/2,H-mb+40),
                cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(0,0,0),1);
    cv::putText(img,title,
                cv::Point(10,(mt+H-mb)/2),
                cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(0,0,0),1);

    cv::imwrite(png_filename, img);
}


void LBmethod::InitDebugDump(const std::string& filename) {
    // Apri in trunc mode
    debug_file.open(filename, std::ios::out);
    if (!debug_file.is_open()) {
        std::cerr << "Errore: non posso aprire " << filename << " per debug dump\n";
    }
    // Facoltativo: intestazione
    debug_file << "# Debug dump per LBmethod\n";
}

void LBmethod::CloseDebugDump() {
    if (debug_file.is_open()) {
        debug_file.close();
    }
}

void LBmethod::DumpGridStateReadable(size_t step, const std::string& stage) {
    if (!debug_file.is_open()) return;

    // Header per questo dump
    debug_file << "step = " << step << "\n";
    debug_file << "stage = " << stage << "\n";

    const int precision = 6; // numero di decimali per i valori

    // 1) ux_e
    debug_file << "ux_e\n";
    for (int jj = static_cast<int>(NY) - 1; jj >= 0; --jj) {
        for (size_t ii = 0; ii < NX; ++ii) {
            size_t idx = INDEX(ii, jj);
            double v = ux_e[idx];
            debug_file << std::scientific << std::setprecision(precision) << v;
            if (ii + 1 < NX) debug_file << ",";
        }
        debug_file << "\n";
    }
    // 2) uy_e
    debug_file << "uy_e\n";
    for (int jj = static_cast<int>(NY) - 1; jj >= 0; --jj) {
        for (size_t ii = 0; ii < NX; ++ii) {
            size_t idx = INDEX(ii, jj);
            double v = uy_e[idx];
            debug_file << std::scientific << std::setprecision(precision) << v;
            if (ii + 1 < NX) debug_file << ",";
        }
        debug_file << "\n";
    }
    // 3) ux_i
    debug_file << "ux_i\n";
    for (int jj = static_cast<int>(NY) - 1; jj >= 0; --jj) {
        for (size_t ii = 0; ii < NX; ++ii) {
            size_t idx = INDEX(ii, jj);
            double v = ux_i[idx];
            debug_file << std::scientific << std::setprecision(precision) << v;
            if (ii + 1 < NX) debug_file << ",";
        }
        debug_file << "\n";
    }
    // 4) uy_i
    debug_file << "uy_i\n";
    for (int jj = static_cast<int>(NY) - 1; jj >= 0; --jj) {
        for (size_t ii = 0; ii < NX; ++ii) {
            size_t idx = INDEX(ii, jj);
            double v = uy_i[idx];
            debug_file << std::scientific << std::setprecision(precision) << v;
            if (ii + 1 < NX) debug_file << ",";
        }
        debug_file << "\n";
    }
    // 3) ux_n
    debug_file << "ux_n\n";
    for (int jj = static_cast<int>(NY) - 1; jj >= 0; --jj) {
        for (size_t ii = 0; ii < NX; ++ii) {
            size_t idx = INDEX(ii, jj);
            double v = ux_n[idx];
            debug_file << std::scientific << std::setprecision(precision) << v;
            if (ii + 1 < NX) debug_file << ",";
        }
        debug_file << "\n";
    }
    // 4) uy_n
    debug_file << "uy_n\n";
    for (int jj = static_cast<int>(NY) - 1; jj >= 0; --jj) {
        for (size_t ii = 0; ii < NX; ++ii) {
            size_t idx = INDEX(ii, jj);
            double v = uy_n[idx];
            debug_file << std::scientific << std::setprecision(precision) << v;
            if (ii + 1 < NX) debug_file << ",";
        }
        debug_file << "\n";
    }
    // 5) rho_q
    debug_file << "rho_q\n";
    for (int jj = static_cast<int>(NY) - 1; jj >= 0; --jj) {
        for (size_t ii = 0; ii < NX; ++ii) {
            size_t idx = INDEX(ii, jj);
            double v = rho_q[idx];
            debug_file << std::scientific << std::setprecision(precision) << v;
            if (ii + 1 < NX) debug_file << ",";
        }
        debug_file << "\n";
    }
    // 6) rho_e
    debug_file << "rho_e\n";
    for (int jj = static_cast<int>(NY) - 1; jj >= 0; --jj) {
        for (size_t ii = 0; ii < NX; ++ii) {
            size_t idx = INDEX(ii, jj);
            double v = rho_e[idx];
            debug_file << std::scientific << std::setprecision(precision) << v;
            if (ii + 1 < NX) debug_file << ",";
        }
        debug_file << "\n";
    }
    // 7) rho_i
    debug_file << "rho_i\n";
    for (int jj = static_cast<int>(NY) - 1; jj >= 0; --jj) {
        for (size_t ii = 0; ii < NX; ++ii) {
            size_t idx = INDEX(ii, jj);
            double v = rho_i[idx];
            debug_file << std::scientific << std::setprecision(precision) << v;
            if (ii + 1 < NX) debug_file << ",";
        }
        debug_file << "\n";
    }
    // 7) rho_n
    debug_file << "rho_n\n";
    for (int jj = static_cast<int>(NY) - 1; jj >= 0; --jj) {
        for (size_t ii = 0; ii < NX; ++ii) {
            size_t idx = INDEX(ii, jj);
            double v = rho_n[idx];
            debug_file << std::scientific << std::setprecision(precision) << v;
            if (ii + 1 < NX) debug_file << ",";
        }
        debug_file << "\n";
    }
    // 8) Ex
    debug_file << "Ex\n";
    for (int jj = static_cast<int>(NY) - 1; jj >= 0; --jj) {
        for (size_t ii = 0; ii < NX; ++ii) {
            size_t idx = INDEX(ii, jj);
            double v = Ex[idx];
            debug_file << std::scientific << std::setprecision(precision) << v;
            if (ii + 1 < NX) debug_file << ",";
        }
        debug_file << "\n";
    }
    // 8) Ey
    debug_file << "Ex\n";
    for (int jj = static_cast<int>(NY) - 1; jj >= 0; --jj) {
        for (size_t ii = 0; ii < NX; ++ii) {
            size_t idx = INDEX(ii, jj);
            double v = Ey[idx];
            debug_file << std::scientific << std::setprecision(precision) << v;
            if (ii + 1 < NX) debug_file << ",";
        }
        debug_file << "\n";
    }

    // Funzione lambda per stampare le distribuzioni f per una specie
    auto dump_f = [&](const std::string& label,
                      const std::vector<double>& f_array)
    {
        debug_file << label << "\n";
        // Direzioni in ordine di “stencil 3x3”:
        static const int dir3x3[3][3] = {{6,2,5},{3,0,1},{7,4,8}};
        debug_file << "directions arrangement:\n";
        for (int rr = 0; rr < 3; ++rr) {
            for (int cc = 0; cc < 3; ++cc) {
                int d = dir3x3[rr][cc];
                debug_file << d;
                if (cc < 2) debug_file << ",";
            }
            debug_file << "\n";
        }
        // Per ciascuna direzione in quell’ordine, dump della griglia
        for (int rr = 0; rr < 3; ++rr) {
            for (int cc = 0; cc < 3; ++cc) {
                int d = dir3x3[rr][cc];
                // Ottieni (cx,cy) se vuoi stampare:
                int cx_d = LBmethod::cx[d];
                int cy_d = LBmethod::cy[d];
                debug_file << label << " dir " << d
                           << " (cx=" << cx_d << ",cy=" << cy_d << ")\n";
                // Ora la griglia: y da NY-1 a 0, x da 0 a NX-1
                for (int jj = static_cast<int>(NY) - 1; jj >= 0; --jj) {
                    for (size_t ii = 0; ii < NX; ++ii) {
                        size_t idx3 = INDEX(ii, jj, d);
                        double v = f_array[idx3];
                        debug_file << std::scientific << std::setprecision(precision) << v;
                        if (ii + 1 < NX) debug_file << ",";
                    }
                    debug_file << "\n";
                }
            }
        }
    };

    // 8) f_e
    dump_f("f_e", f_e);
    // 9) f_i
    dump_f("f_i", f_i);
    // 9) f_n
    dump_f("f_n", f_n);
    // 10) f_eq_e
    dump_f("f_eq_e", f_eq_e);
    // 11) f_eq_i
    dump_f("f_eq_i", f_eq_i);
    // 10) f_eq_n
    dump_f("f_eq_n", f_eq_n);
    // 11) f_eq_e_i
    dump_f("f_eq_e_i", f_eq_e_i);
    // 11) f_eq_i_e
    dump_f("f_eq_i_e", f_eq_i_e);
    // 11) f_eq_e_n
    dump_f("f_eq_e_n", f_eq_e_n);
    // 11) f_eq_n_e
    dump_f("f_eq_n_e", f_eq_n_e);
    // 11) f_eq_i_n
    dump_f("f_eq_i_n", f_eq_i_n);
    // 11) f_eq_n_i
    dump_f("f_eq_n_i", f_eq_n_i);


    // Riga vuota come separatore
    debug_file << "\n";
    // Flush per sicurezza (moltissimi dati)
    debug_file.flush();
}
