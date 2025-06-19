#include "plasma.hpp"
#include "utils.hpp"

#include <cmath>
#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

//──────────────────────────────────────────────────────────────────────────────
//  Static member definitions for D2Q9:
//──────────────────────────────────────────────────────────────────────────────
const std::array<int, Q> LBmethod::cx = { { 0, 1, 0, -1,  0,  1, -1, -1,  1 } };
const std::array<int, Q> LBmethod::cy = { { 0, 0, 1,  0, -1,  1,  1, -1, -1 } };
const std::array<double, Q> LBmethod::w = { {
    4.0/9.0,
    1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0
} };


//──────────────────────────────────────────────────────────────────────────────
//  Constructor: everything passed in SI → convert to lattice units
//──────────────────────────────────────────────────────────────────────────────
LBmethod::LBmethod(const int    _NSTEPS,
                   const int    _NX,
                   const int    _NY,
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
                   const poisson::PoissonType _poisson_type,
                   const streaming::BCType      _bc_type,
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
    for (int x = 0; x < NX; ++x) {
        for(int y = 0; y < NY; ++y){
            for (int i=0;i<Q;++i){
                const int idx_3 = INDEX(x, y, i,NX,Q);

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
        for (int x = 0; x < NX; ++x) {
            for (int y = 0; y < NY; ++y) {
                const int idx = INDEX(x, y, NX); // Get 1D index for 2D point (x, y)
                const double u2_e = ux_e[idx] * ux_e[idx] + uy_e[idx] * uy_e[idx]; // Square of the speed magnitude
                const double u2_i = ux_i[idx] * ux_i[idx] + uy_i[idx] * uy_i[idx]; 
                const double u2_n = ux_n[idx] * ux_n[idx] + uy_n[idx] * uy_n[idx]; 
                const double u2_e_i = ux_e_i[idx] * ux_e_i[idx] + uy_e_i[idx] * uy_e_i[idx]; // Square of the speed magnitude for electron-ion interaction
                const double u2_e_n = ux_e_n[idx] * ux_e_n[idx] + uy_e_n[idx] * uy_e_n[idx];
                const double u2_i_n = ux_i_n[idx] * ux_i_n[idx] + uy_i_n[idx] * uy_i_n[idx];
                const double den_e = rho_e[idx]; // Electron density
                const double den_i = rho_i[idx]; // Ion density
                const double den_n = rho_n[idx]; // Neutrals density
                
                for (int i = 0; i < Q; ++i) {
                    const int idx_3=INDEX(x, y, i,NX,Q);
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
        for (int x=0; x<NX; ++x){
            for (int y = 0; y < NY; ++y) {
                const int idx = INDEX(x, y, NX);
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

                for (int i = 0; i < Q; ++i) {
                    const int idx_3 = INDEX(x, y, i,NX,Q);
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
                        ux_e[idx]=0.0;
                    else
                        ux_e[idx] = ux_loc_e / rho_loc_e;
                    if(uy_loc_e==rho_loc_e || uy_loc_e==-rho_loc_e)
                        uy_e[idx]=0.0;
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
                        ux_i[idx]=0.0;
                    else
                        ux_i[idx] = ux_loc_i / rho_loc_i;
                    if(uy_loc_i==rho_loc_i || uy_loc_i==-rho_loc_i)
                        uy_i[idx]=0.0;
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

void LBmethod::Run_simulation() {
    
    // Set threads for this simulation
    omp_set_num_threads(n_cores);

    // Initialize visualize stuff
    visualize::InitVisualization(NX, NY, NSTEPS);

    //──────────────────────────────────────────────────────────────────────────────
    //  Main loop: for t = 0 … NSTEPS−1,
    //    [1] Update macros (ρ, u)
    //    [2] Calculate equilibrium distribution functions
    //    [3] Collisions (BGK + forcing)
    //    [4] Streaming (+ BC)
    //    [5] Solve Poisson → update Ex, Ey
    //    [6] Visualization
    //──────────────────────────────────────────────────────────────────────────────
    for (int t=0; t<NSTEPS; ++t){
        // Macroscopic update:  ρ = Σ_i f_i,   ρ u = Σ_i f_i c_i + ½ F  T=Σ_i g_i
        UpdateMacro(); 
        ComputeEquilibrium();

        // g(x,y,t)_postcoll=g(x,y,t) + (g_eq - g)/tau + Source
        // f(x,y,t)_postcoll=f(x,y,t) + (f_eq - f)/tau + dt*F
        collisions::Collide(g_e, g_i, g_n, g_eq_e, g_eq_i, g_eq_n,
                            g_eq_e_i, g_eq_e_n, g_eq_i_n, g_eq_i_e, g_eq_n_e, g_eq_n_i,
                            f_e, f_i, f_n, f_eq_e, f_eq_i, f_eq_n,
                            f_eq_e_i, f_eq_e_n, f_eq_i_n, f_eq_i_e, f_eq_n_e, f_eq_n_i,
                            rho_e, rho_i, rho_n,
                            ux_e, uy_e,
                            ux_i, uy_i,
                            ux_n, uy_n,
                            Ex, Ey,
                            q_e, q_i,
                            m_e, m_i,
                            f_temp_e, f_temp_i, f_temp_n,
                            cx, cy, w,
                            NX, NY, Kb, cs2); 

        // f(x+cx,y+cx,t+1)=f(x,y,t)
        // +BC applyed
        streaming::Stream(f_e, f_i, f_n, 
                          f_temp_e, f_temp_i, f_temp_n,
                          g_e, g_i, g_n,
                          cx, cy,
                          NX, NY, bc_type);
        // Solve the poisson equation with the method chosen
        // Also BCs are important
        poisson::SolvePoisson(Ex,
                              Ey,
                              rho_q,
                              NX, NY,
                              omega_sor,
                              poisson_type,
                              bc_type);
    
        // Update video and data for plot
        visualize::UpdateVisualization(t, NX, NY,
                                       ux_e, uy_e,
                                       ux_i, uy_i,
                                       ux_n, uy_n,
                                       T_e, T_i, T_n,
                                       rho_e, rho_i, rho_n,
                                       rho_q, Ex, Ey);
    }
    //Close Visualize stuff
    visualize::CloseVisualization();


    std::cout << "Simulation ended " << std::endl;
}
