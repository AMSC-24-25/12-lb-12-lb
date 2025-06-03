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
    f_eq_e_i.assign(NX * NY * Q, 0.0);
    f_eq_i_e.assign(NX * NY * Q, 0.0);

    n_e.assign(NX * NY, 1.0);
    n_i.assign(NX * NY, 1.0);
    ux_e.assign(NX * NY, 0.0);
    uy_e.assign(NX * NY, 0.0);
    ux_i.assign(NX * NY, 0.0);
    uy_i.assign(NX * NY, 0.0);
    ux_e_i.assign(NX * NY, 0.0);
    uy_e_i.assign(NX * NY, 0.0);

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
        << "v_th_e = "<<std::sqrt(kB_SI*T_e_SI/m_e_SI) << " m/s v_th_i = "<< std::sqrt(kB_SI*T_i_SI/m_i_SI)<< " m/s\n"
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
    }
}
//──────────────────────────────────────────────────────────────────────────────
//  Compute D2Q9 equilibrium for a given (ρ, ux, uy) and sound‐speed cs2.
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::computeFeq() {//It's the same for all the species, maybe can be used as a function
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
                }
            }
        }
}
//──────────────────────────────────────────────────────────────────────────────
//  Update macroscopic variables for both species:
//    ρ = Σ_i f_i,
//    ρ u = Σ_i f_i c_i + (1/2)*F
//  where F = qom_latt * (Ex_cell, Ey_cell).
//──────────────────────────────────────────────────────────────────────────────
void LBmethod::UpdateMacro() {
    //Find a way to evaluete the temperature I don't know how to do it
    //maybe from another distribution function g that has to be imposed and solved
    double n_loc_e = 0.0;
    double ux_loc_e = 0.0;
    double uy_loc_e = 0.0;

    double n_loc_i = 0.0;
    double ux_loc_i = 0.0;
    double uy_loc_i = 0.0;

    //const double sigma_el_ion=M_PI*(r_el+r_ion)*(r_el+r_ion); //Cross section for electron-ion interaction
    //const double mean_vel_el = std::sqrt(8*kb/(m_el*M_PI)); //Mean velocity of electrons without Temperature dependence
    //const double mean_vel_ion = std::sqrt(8*kb/(A_ion*m_p*M_PI)); //Mean velocity of ions without Temperature dependence
    //const double mass_el_ion = m_el * A_ion * m_p /(m_el + A_ion * m_p); //Reduced mass for electron-ion interaction
    #pragma omp parallel for collapse(2) private(n_loc_e, ux_loc_e, uy_loc_e, n_loc_i, ux_loc_i, uy_loc_i)
        for (size_t x=0; x<NX; ++x){
            for (size_t y = 0; y < NY; ++y) {
                const size_t idx = INDEX(x, y);
                n_loc_i = 0.0;
                ux_loc_i = 0.0;
                uy_loc_i = 0.0;

                n_loc_e = 0.0;
                ux_loc_e = 0.0;
                uy_loc_e = 0.0;


                for (size_t i = 0; i < Q; ++i) {
                    const size_t idx_3 = INDEX(x, y, i);
                    const double fi_e=f_e[idx_3];
                    n_loc_e += fi_e;
                    ux_loc_e += fi_e * cx[i];
                    uy_loc_e += fi_e * cy[i];

                    const double fi_i=f_i[idx_3];
                    n_loc_i += fi_i;
                    ux_loc_i += fi_i * cx[i];
                    uy_loc_i += fi_i * cy[i];
                    
                }
                if (n_loc_e<1e-10){
                    n_e[idx] = 0.0;
                    ux_e[idx] = 0.0;
                    uy_e[idx] = 0.0;
                }else if (n_loc_i<1e-10){
                    n_i[idx] = 0.0;
                    ux_i[idx] = 0.0;
                    uy_i[idx] = 0.0;
                }
                else {
                    n_e[idx] = n_loc_e;
                    n_i[idx] = n_loc_i;

                    //Add the force term to velocity
                    ux_loc_e+=0.5*(qom_e_latt*Ex[idx]);
                    uy_loc_e+=0.5*(qom_e_latt*Ey[idx]);
                    ux_loc_i+=0.5*(qom_i_latt*Ex[idx]);
                    uy_loc_i+=0.5*(qom_i_latt*Ey[idx]);

                    //assign velocity
                    ux_e[idx]=ux_loc_e/n_loc_e;
                    uy_e[idx]=uy_loc_e/n_loc_e;
                    ux_i[idx]=ux_loc_i/n_loc_i;
                    uy_i[idx]=uy_loc_i/n_loc_i;

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
    
    computeFeq();
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
                );
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

    f_i.swap(f_temp_i);
    f_e.swap(f_temp_e);
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
    #pragma omp parallel
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
                        f_temp_i[INDEX(x_str, y_str, i)] = f_i[INDEX(x, y, i)];
                        f_temp_e[INDEX(x_str, y_str, i)] = f_e[INDEX(x, y, i)];
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

void LBmethod::Run_simulation() {
    
    // Set threads for this simulation
    omp_set_num_threads(n_cores);


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

    //──────────────────────────────────────────────────────────────────────────────
    //  Main loop: for t = 0 … NSTEPS−1,
    //    [1] Update macros (ρ, u)
    //    [2] Solve Poisson → update Ex, Ey
    //    [3] Collisions (BGK + forcing)
    //    [4] Streaming (+ BC)
    //    [5] Visualization
    //──────────────────────────────────────────────────────────────────────────────
    for (size_t t=0; t<NSTEPS; ++t){
        UpdateMacro(); // rho=sum(f), ux=sum(f*c_x)/rho, uy=sum(f*c_y)/rho

        if (t%100==0) {
            auto max_ne = std::max_element(n_e.begin(), n_e.end());
            auto max_ni = std::max_element(n_i.begin(), n_i.end());
            auto max_uxe = std::max_element(ux_e.begin(), ux_e.end());
            auto max_uye = std::max_element(uy_e.begin(), uy_e.end());
            auto max_uxi = std::max_element(ux_i.begin(), ux_i.end());
            auto max_uyi = std::max_element(uy_i.begin(), uy_i.end());
            auto max_Ex = std::max_element(Ex.begin(), Ex.end());
            auto max_Ey = std::max_element(Ey.begin(), Ey.end());
            auto max_rho = std::max_element(rho_q_latt.begin(), rho_q_latt.end());
            std::cout <<"Step:"<<t<<std::endl;
            std::cout <<"max n_e= "<<*max_ne<<std::endl;
            std::cout <<"max n_i= "<<*max_ni<<std::endl;
            std::cout <<"max ux_e= "<<*max_uxe<<std::endl;
            std::cout <<"max uy_e= "<<*max_uye<<std::endl;
            std::cout <<"max ux_i= "<<*max_uxi<<std::endl;
            std::cout <<"max uy_i= "<<*max_uyi<<std::endl;
            std::cout <<"max Ex= "<<*max_Ex<<std::endl;
            std::cout <<"max Ey= "<<*max_Ey<<std::endl;
            std::cout <<"max rho_q (latt)= "<<*max_rho<<std::endl;
            std::cout <<std::endl;
        }

        SolvePoisson();

        

        Collisions(); // f(x,y,t+1)=f(x-cx,y-cy,t) + tau * (f_eq - f) + dt*F
        Streaming(); // f(x,y,t+1)=f(x-cx,y-cy,t)
        Visualization(t);
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
    }  


    video_writer.release();
    std::cout << "Video saved as " << video_filename << std::endl;
}
//──────────────────────────────────────────────────────────────────────────────
//  Visualization stub.  Use OpenCV to save density images at time t, etc.
//──────────────────────────────────────────────────────────────────────────────
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
            const size_t idx = INDEX(x, y);

            const double ux_i_loc = ux_i[idx], uy_i_loc = uy_i[idx];
            const double ux_e_loc = ux_e[idx], uy_e_loc = uy_e[idx];
            const double n_i_loc = n_i[idx], n_e_loc = n_e[idx];

            velocity_magn_mat_ion.at<float>(y, x) = std::hypot(ux_i_loc, uy_i_loc);
            velocity_magn_mat_el.at<float>(y, x) = std::hypot(ux_e_loc, uy_e_loc);
            density_mat_ion.at<float>(y, x) = static_cast<float>(n_i_loc);
            density_mat_el.at<float>(y, x) = static_cast<float>(n_e_loc);

            combined_density_mat.at<float>(y, x) = static_cast<float>(n_i_loc - n_e_loc);
            combined_velocity_mat.at<float>(y, x) = std::hypot(ux_i_loc - ux_e_loc, uy_i_loc - uy_e_loc);
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
    cv::hconcat(std::vector<cv::Mat>{wrap_with_label(density_heatmap_ion, "n Ions"), wrap_with_label(density_heatmap_el, "n Electrons"), wrap_with_label(combined_density_heatmap, "Comb rho")}, top_row);
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
