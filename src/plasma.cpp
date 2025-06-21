#include "plasma.hpp"

#include <cmath>
#include <omp.h>
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
//  Constructor
//──────────────────────────────────────────────────────────────────────────────
LBmethod::LBmethod(const int    _NSTEPS,
                   const int    _NX,
                   const int    _NY,
                   const size_t    _n_cores,
                   const int    _Z_ion,
                   const int    _A_ion,
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
{
    //──────────────────────────────────────────────────────────────────────────────
    //  Allocate all the vectors
    //──────────────────────────────────────────────────────────────────────────────
    const int size_distr = NX * NY *Q;

    f_e.assign(size_distr, 0.0); //in order to allow deciding initial conditions this has to be set to zero
    f_i.assign(size_distr, 0.0); //in this way we can define the function only where we want it
    f_n.assign(size_distr, 0.0);
    
    f_eq_e.resize(size_distr);
    f_eq_i.resize(size_distr);
    f_eq_n.resize(size_distr);
    
    f_eq_e_i.resize(size_distr);
    f_eq_i_e.resize(size_distr);
    f_eq_e_n.resize(size_distr);
    f_eq_n_e.resize(size_distr);
    f_eq_i_n.resize(size_distr);
    f_eq_n_i.resize(size_distr);

    g_e.assign(size_distr, 0.0);
    g_i.assign(size_distr, 0.0);
    g_n.assign(size_distr, 0.0);

    g_eq_e.resize(size_distr);
    g_eq_i.resize(size_distr);
    g_eq_n.resize(size_distr);

    g_eq_e_i.resize(size_distr);
    g_eq_i_e.resize(size_distr);
    g_eq_e_n.resize(size_distr);
    g_eq_n_e.resize(size_distr);
    g_eq_i_n.resize(size_distr);
    g_eq_n_i.resize(size_distr);

    temp_e.resize(size_distr);
    temp_i.resize(size_distr);
    temp_n.resize(size_distr);

    const int size_macro = NX * NY;

    rho_e.resize(size_macro);
    rho_i.resize(size_macro);
    rho_n.resize(size_macro);
    ux_e.resize(size_macro);
    uy_e.resize(size_macro);
    ux_i.resize(size_macro);
    uy_i.resize(size_macro);
    ux_n.resize(size_macro);
    uy_n.resize(size_macro);
    ux_e_i.resize(size_macro);
    uy_e_i.resize(size_macro);
    ux_e_n.resize(size_macro);
    uy_e_n.resize(size_macro);
    ux_i_n.resize(size_macro);
    uy_i_n.resize(size_macro);

    T_e.resize(size_macro);
    T_i.resize(size_macro);
    T_n.resize(size_macro);
    // Pulsed initial eleectric field
    Ex.assign(size_macro, Ex_ext);
    Ey.assign(size_macro, Ey_ext);

    // In lattice units
    rho_q.resize(size_macro);

    // Initialize fields:  set f = w * m and g = w * T
    Initialize();
}

//──────────────────────────────────────────────────────────────────────────────
//  Impose initial conditions chosen
//  No need to initialize the macroscopic quantities 
//  since UpdateMacro will do the work
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::Initialize() {
    // Assign electrons andd ions only in the center
    #pragma omp parallel for collapse(3) schedule(static)
    for (int x = (NX/4)+1; x <(3*NX/4); ++x) {
        for(int y = (NY/4)+1; y <(3*NY/4); ++y){
            for (int i=0;i<Q;++i){
                const int idx_3 = INDEX(x, y, i,NX,Q);
                const double weight = w[i];
                f_e[idx_3] =  weight * rho_e_init; // Equilibrium function for electrons
                g_e[idx_3] =  weight * T_e_init; // Thermal function for electrons
                f_i[idx_3] =  weight * rho_i_init; // Equilibrium function for ions
                g_i[idx_3] =  weight * T_i_init; // Thermal function for ions
            }
        }
    }
    //Assign the neutrals to all the zone
    #pragma omp parallel for collapse(3) schedule(static)
    for (int x = 0; x < NX; ++x) {
        for(int y = 0; y < NY; ++y){
            for (int i=0;i<Q;++i){
                const int idx_3 = INDEX(x, y, i,NX,Q);
                const double weight = w[i];
                f_n[idx_3] =  weight * rho_n_init;
                g_n[idx_3] =  weight * T_n_init;
            }
        }
    }
}
//──────────────────────────────────────────────────────────────────────────────
//  Compute D2Q9 equilibrium for f_eq and g_eq
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::ComputeEquilibrium() {
    // Compute the equilibrium distribution functions
    const double invcs2=1.0/cs2;
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
                const double Te = T_e[idx]; //Electron temperature
                const double Ti = T_i[idx]; //Ion temperature
                const double Tn = T_n[idx]; //Neutrals temperature
                

                for (int i = 0; i < Q; ++i) {
                    const int idx_3=INDEX(x, y, i,NX,Q);
                    const double cu_e = cx[i]*ux_e[idx] +cy[i]*uy_e[idx]; // Dot product (c_i · u)
                    const double cu_i = cx[i]*ux_i[idx] +cy[i]*uy_i[idx];
                    const double cu_n = cx[i]*ux_n[idx] +cy[i]*uy_n[idx];
                    const double cu_e_i = cx[i]*ux_e_i[idx] +cy[i]*uy_e_i[idx]; // Dot product (c_i · u) for electron-ion interaction
                    const double cu_e_n = cx[i]*ux_e_n[idx] +cy[i]*uy_e_n[idx];
                    const double cu_i_n = cx[i]*ux_i_n[idx] +cy[i]*uy_i_n[idx];
                    const double weight = w[i];


                    // Compute f_eq from discretization of Maxwell Boltzmann distribution function
                    f_eq_e[idx_3]= weight*den_e*(
                        1.0 +
                        (cu_e) *invcs2+
                        (cu_e * cu_e) * 0.5 *invcs2 *invcs2-
                        u2_e * 0.5*invcs2
                    );
                    f_eq_i[idx_3]= weight*den_i*(
                        1.0 +
                        (cu_i) *invcs2+
                        (cu_i * cu_i) * 0.5 *invcs2*invcs2 -
                        u2_i * 0.5*invcs2
                    );
                    f_eq_n[idx_3]= weight*den_n*(
                        1.0 +
                        (cu_n)*invcs2 +
                        (cu_n * cu_n) * 0.5 *invcs2*invcs2 -
                        u2_n * 0.5*invcs2
                    );

                    f_eq_e_i[idx_3]= weight*den_e*(
                        1.0 +
                        (cu_e_i)*invcs2 +
                        (cu_e_i * cu_e_i) * 0.5 *invcs2*invcs2 -
                        u2_e_i * 0.5*invcs2
                    );
                    f_eq_i_e[idx_3]= weight*den_i*(
                        1.0 +
                        (cu_e_i)*invcs2 +
                        (cu_e_i * cu_e_i) * 0.5 *invcs2*invcs2 -
                        u2_e_i * 0.5*invcs2
                    );
                    f_eq_e_n[idx_3]= weight*den_e*(
                        1.0 +
                        (cu_e_n) *invcs2+
                        (cu_e_n * cu_e_n) * 0.5 *invcs2*invcs2 -
                        u2_e_n * 0.5*invcs2
                    );
                    f_eq_n_e[idx_3]= weight*den_n*(
                        1.0 +
                        (cu_e_n)*invcs2 +
                        (cu_e_n * cu_e_n) * 0.5 *invcs2*invcs2 -
                        u2_e_n * 0.5 *invcs2
                    );
                    f_eq_i_n[idx_3]= weight*den_i*(
                        1.0 +
                        (cu_i_n) *invcs2+
                        (cu_i_n * cu_i_n) * 0.5 *invcs2 *invcs2-
                        u2_i_n * 0.5 *invcs2
                    );
                    f_eq_n_i[idx_3]= weight*den_n*(
                        1.0 +
                        (cu_i_n)*invcs2 +
                        (cu_i_n * cu_i_n) * 0.5 *invcs2*invcs2 -
                        u2_i_n * 0.5*invcs2
                    );

                    g_eq_e[idx_3]=weight*Te*(
                        1.0 +
                        (cu_e)*invcs2 +
                        (cu_e * cu_e ) * 0.5 *invcs2*invcs2 -
                        u2_e * 0.5*invcs2
                    );
                    g_eq_i[idx_3]=weight*Ti*(
                        1.0 +
                        (cu_i)*invcs2 +
                        (cu_i * cu_i ) * 0.5 *invcs2 *invcs2 -
                        u2_i * 0.5*invcs2
                    );
                    g_eq_n[idx_3]=weight*Tn*(
                        1.0 +
                        (cu_n)*invcs2 +
                        (cu_n * cu_n ) * 0.5 *invcs2 *invcs2-
                        u2_n * 0.5*invcs2
                    );
                    g_eq_e_i[idx_3]= weight*Te*(
                        1.0 +
                        (cu_e_i)*invcs2 +
                        (cu_e_i * cu_e_i) * 0.5 *invcs2 *invcs2 -
                        u2_e_i * 0.5 *invcs2
                    );
                    g_eq_i_e[idx_3]= weight*Ti*(
                        1.0 +
                        (cu_e_i) *invcs2+
                        (cu_e_i * cu_e_i) * 0.5 *invcs2 *invcs2-
                        u2_e_i * 0.5 *invcs2
                    );
                    g_eq_e_n[idx_3]= weight*Te*(
                        1.0 +
                        (cu_e_n) *invcs2+
                        (cu_e_n * cu_e_n) * 0.5 *invcs2 *invcs2-
                        u2_e_n * 0.5 *invcs2
                    );
                    g_eq_n_e[idx_3]= weight*Tn*(
                        1.0 +
                        (cu_e_n)*invcs2 +
                        (cu_e_n * cu_e_n) * 0.5 *invcs2 *invcs2 -
                        u2_e_n * 0.5 *invcs2
                    );
                    g_eq_i_n[idx_3]= weight*Ti*(
                        1.0 +
                        (cu_i_n)*invcs2 +
                        (cu_i_n * cu_i_n) * 0.5 *invcs2 *invcs2-
                        u2_i_n * 0.5*invcs2
                    );
                    g_eq_n_i[idx_3]= weight*Tn*(
                        1.0 +
                        (cu_i_n)*invcs2 +
                        (cu_i_n * cu_i_n) * 0.5 *invcs2*invcs2 -
                        u2_i_n * 0.5 *invcs2
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
                    if(ux_loc_e==rho_loc_e || ux_loc_e==-rho_loc_e) // In order to avoid exiting conditions of the model when the initial streaming is performed
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
                
                // Lattice‐unit charge density:
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
                            temp_e, temp_i, temp_n,
                            cx, cy, w,
                            NX, NY, Kb, cs2); 

        // f(x+cx,y+cx,t+1)=f(x,y,t)
        // +BC applyed
        streaming::Stream(f_e, f_i, f_n, 
                          temp_e, temp_i, temp_n,
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
