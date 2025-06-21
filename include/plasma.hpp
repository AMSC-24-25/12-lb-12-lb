#pragma once

#include "collisions.hpp"
#include "poisson.hpp"
#include "streaming.hpp"
#include "utils.hpp"
#include "visualize.hpp"

#include <array>
#include <vector>

//--------------------------------------------------------------------------------
// LBmethod: performs a two‐species (electron + ion) D2Q9 LBM under an electric field.
// All “physical” parameters are passed in SI units to the constructor, and inside
// the constructor they are converted to lattice units.  After that, Run_simulation()
// can be called to execute the time loop.
//--------------------------------------------------------------------------------
class LBmethod {
public:
    // Constructor: pass in *all* physical/SI parameters + grid‐size + time‐steps
    //
    //   NSTEPS       : number of time steps to run
    //   NX, NY       : number of lattice nodes in x and y (grid size)
    //   n_cores      : number of OpenMP threads (optional, can be ignored)
    //   Z_ion, A_ion : ionic charge‐number and mass‐number (for computing ion mass)
    //
    //   T_e_SI, T_i_SI : electron and ion temperatures [K]
    //   Ex_SI, Ey_SI   : uniform external E‐field [V/m] (can be overridden by Poisson solver)
    //
    //   poisson_type   : which Poisson solver to use (NONE, GAUSS_SEIDEL, or SOR)
    //   bc_type        : which streaming/BC to use (PERIODIC or BOUNCE_BACK)
    //   omega_sor      : over‐relaxation factor for SOR (only used if poisson_type==SOR)
    //
    LBmethod(const int    NSTEPS,
             const int    NX,
             const int    NY,
             const size_t    n_cores,
             const int    Z_ion,
             const int    A_ion,
             const double    Ex_SI,
             const double    Ey_SI,
             const double    T_e_SI_init,
             const double    T_i_SI_init,
             const double    T_n_SI_init,
             const double    n_e_SI_init,
             const double    n_n_SI_init,
             const poisson::PoissonType poisson_type,
             const streaming::BCType      bc_type,
             const double    omega_sor);

    // Run the complete simulation
    void Run_simulation();


private:
    //──────────────────────────────────────────────────────────────────────────────
    // 1) “Raw” (SI) Inputs
    //──────────────────────────────────────────────────────────────────────────────
    const int  NSTEPS;       // total number of time steps
    const int  NX, NY;       // grid dimensions
    const size_t  n_cores;      // # of OpenMP threads (optional)
    const int  Z_ion;        // ionic atomic number (e.g. Z=1 for H+)
    const int  A_ion;        // ionic mass # (e.g. A=1 for H+)
    const double    Ex_SI;
    const double    Ey_SI;
    const double    T_e_SI_init;
    const double    T_i_SI_init;
    const double    T_n_SI_init;
    const double    n_e_SI_init;
    const double    n_n_SI_init;
    const poisson::PoissonType  poisson_type; // which Poisson solver to run
    const streaming::BCType       bc_type;      // which streaming/BC we use
    const double       omega_sor;    // over‐relaxation factor for SOR

    //──────────────────────────────────────────────────────────────────────────────
    // 2) Physical Constants (SI)
    //──────────────────────────────────────────────────────────────────────────────
    static constexpr double kB_SI       = 1.380649e-23;   // [J/K]
    static constexpr double e_charge_SI = 1.602176634e-19;// [C]
    static constexpr double epsilon0_SI = 8.854187817e-12;// [F/m]
    static constexpr double m_e_SI      = 9.10938356e-31; // [kg]
    static constexpr double u_SI        = 1.66053906660e-27; // [kg]
    static constexpr double m_p_SI      = 1.67262192595e-27; // [kg]
    static constexpr double m_ne_SI     = 1.67492749804e-27; // [kg]

    const double m_i_SI = A_ion * u_SI; //[kg]
    const double m_n_SI = A_ion * u_SI; //[kg]
    //──────────────────────────────────────────────────────────────────────────────
    // Conversion of quantities from SI to LU:
    //──────────────────────────────────────────────────────────────────────────────
    const double n0_SI = n_e_SI_init;

    const double M0_SI = m_e_SI; // physical mass [kg]
    const double T0_SI = T_e_SI_init; // physical temperature [K]
    const double Q0_SI = e_charge_SI; // physical charge [C]
    const double L0_SI = std::sqrt(epsilon0_SI * kB_SI * T0_SI / (n0_SI * Q0_SI * Q0_SI))*1e-2; // physical lenght = lambda_D/100 [m]
    const double t0_SI = std::sqrt(epsilon0_SI * M0_SI / (3.0 * n0_SI * Q0_SI * Q0_SI))  *1e-2; // physical time = rad(3)/w_p/100 [s]
    
    //other useful obtained scaling quantities
    const double E0_SI = M0_SI*L0_SI/(Q0_SI*t0_SI*t0_SI); // physical electric field [V/m]
    const double v0_SI = L0_SI / t0_SI; // physical velocity [m/s]
    const double F0_SI = M0_SI * L0_SI / (t0_SI * t0_SI); // physical force [N]

    //──────────────────────────────────────────────────────────────────────────────
    // 3) Lattice‐Unit Quantities rescaled here
    //──────────────────────────────────────────────────────────────────────────────
    // Sound‐speeds in lattice units from D2Q9 c_s^2=1/3
    const double cs2 = kB_SI * T0_SI / M0_SI * t0_SI * t0_SI / (L0_SI * L0_SI);
 
    const double Kb = kB_SI* (t0_SI * t0_SI * T0_SI)/(L0_SI * L0_SI * M0_SI);
    
    // Converted E‐field in lattice units:
    const double Ex_ext = Ex_SI / E0_SI, 
                 Ey_ext = Ey_SI / E0_SI; // external intial E‐field in lattice units

    // Converted temperatures in lattice units:
    const double T_e_init = T_e_SI_init / T0_SI, 
                 T_i_init = T_i_SI_init / T0_SI,
                 T_n_init = T_n_SI_init / T0_SI; // initial temperatures in lattice units

    // mass in lattice units:
    const double m_e = m_e_SI / M0_SI, // electron mass in lattice units
                 m_i = m_i_SI / M0_SI, // ion mass in lattice masses
                 m_n = m_n_SI / M0_SI; // neutrals mass in lattice units

    // Converted charge in lattice units:
    const double q_e = - e_charge_SI / Q0_SI; // electron charge in lattice units
    const double q_i = Z_ion * e_charge_SI / Q0_SI; // ion charge in lattice units

    // Initial density in lattice unit
    const double rho_e_init = m_e * n_e_SI_init / n0_SI, // electron density in lattice units
                 rho_i_init = m_i * n_e_SI_init / n0_SI / Z_ion, // ion density in lattice units. The idea behind /Z_ion is overall neutrality of the plamsa at the start
                 rho_n_init = m_n * n_n_SI_init / n0_SI; // neutrals density in lattice units
    //──────────────────────────────────────────────────────────────────────────────
    // 4) D2Q9 Setup
    //──────────────────────────────────────────────────────────────────────────────
    static const std::array<int, Q> cx; // = {0,1,0,-1,0,1,-1,-1,1};
    static const std::array<int, Q> cy; // = {0,0,1,0,-1,1,1,-1,-1};
    static const std::array<double, Q> w; // weights

    //──────────────────────────────────────────────────────────────────────────────
    // 5) Per‐Node (“lattice‐unit”) Fields
    //──────────────────────────────────────────────────────────────────────────────
    // Distribution functions: f_e[i + Q*(x + NX*y)], f_i[i + Q*(x + NX*y)]
    std::vector<double>   f_e,    f_i,    f_n;    
    // Equilibrium distribution functions
    std::vector<double>   f_eq_e,    f_eq_i,    f_eq_n,   
                          f_eq_e_i,  f_eq_i_e,
                          f_eq_e_n,  f_eq_n_e,
                          f_eq_i_n,  f_eq_n_i;
                         
    // Thermal distribution function
    std::vector<double>   g_e,    g_i,    g_n;   
    // Equilibrium distribution functions
    std::vector<double>   g_eq_e,    g_eq_i,    g_eq_n,
                          g_eq_e_i,  g_eq_i_e,
                          g_eq_e_n,  g_eq_n_e,
                          g_eq_i_n,  g_eq_n_i;
    
    // Themporal distribution functions
    std::vector<double> temp_e, temp_i, temp_n;

    // Macroscopic moments (per cell)
    std::vector<double>   rho_e,  rho_i, rho_n;      // densities
    std::vector<double>   ux_e,   uy_e,       // velocities
                          ux_i,   uy_i,
                          ux_n,   uy_n,
                          ux_e_i, uy_e_i,
                          ux_e_n, uy_e_n,
                          ux_i_n, uy_i_n;
    
    // Temperature vectors
    std::vector<double>  T_e,  T_i,  T_n;

    // Electric potential in lattice units
    std::vector<double>   Ex,    Ey;         // self‐consistent E (overwrites Ex_latt_init)

    // Charge density (per cell in lattice units)
    std::vector<double>   rho_q; // dimensionless (#/cell * e_charge)

    //──────────────────────────────────────────────────────────────────────────────
    // 6) Private Methods
    //──────────────────────────────────────────────────────────────────────────────

    // (a) Initialize all fields (set f = f_eq at t=0, zero φ, set E=Ex_latt_init)
    void Initialize();

    // (b) Compute equilibrium f_eq
    void ComputeEquilibrium();
    
    // (d) Macroscopic update:  ρ = Σ_i f_i,   ρ u = Σ_i f_i c_i + ½ F  T=Σ_i g_i
    void UpdateMacro();
  
};
