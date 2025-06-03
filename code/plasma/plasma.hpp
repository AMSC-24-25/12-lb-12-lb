#ifndef LBM_H
#define LBM_H

#include <vector>
#include <array>
#include <utility>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>        
#include <opencv2/imgproc.hpp>   
#include <opencv2/highgui.hpp> 
#include <opencv2/videoio.hpp> 
#include <filesystem>
#include <fftw3.h>

//--------------------------------------------------------------------------------
// Enumerations for choosing Poisson solver and Streaming/BC type
//--------------------------------------------------------------------------------
enum class PoissonType { NONE = 0, GAUSS_SEIDEL = 1, SOR = 2, FFT = 3 };
enum class BCType      { PERIODIC = 0, BOUNCE_BACK = 1 };
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
    //   r_ion        : ionic radius [m] (unused in this template but stored)
    //
    //   Lx_SI, Ly_SI   : physical domain size in x and y [m]
    //   dt_SI          : physical time‐step [s]
    //   T_e_SI, T_i_SI : electron and ion temperatures [K]
    //   Ex_SI, Ey_SI   : uniform external E‐field [V/m] (can be overridden by Poisson solver)
    //
    //   poisson_type   : which Poisson solver to use (NONE, GAUSS_SEIDEL, or SOR)
    //   bc_type        : which streaming/BC to use (PERIODIC or BOUNCE_BACK)
    //   omega_sor      : over‐relaxation factor for SOR (only used if poisson_type==SOR)
    //
    LBmethod(const size_t    NSTEPS,
             const size_t    NX,
             const size_t    NY,
             const size_t    n_cores,
             const size_t    Z_ion,
             const size_t    A_ion,
             const double    r_ion,
             const double    Lx_SI,
             const double    Ly_SI,
             const double    dt_SI,
             const double    T_e_SI,
             const double    T_i_SI,
             const double    Ex_SI,
             const double    Ey_SI,
             const PoissonType poisson_type,
             const BCType      bc_type,
             const double    omega_sor = 1.8);

    // Run the complete simulation (calls Initialize(), then loops on TimeSteps)
    void Run_simulation();

    // (Optional) visualization hook, can be implemented to write frames out.
    void Visualization(size_t t);
private:
    //──────────────────────────────────────────────────────────────────────────────
    // 1) “Raw” (SI) Inputs
    //──────────────────────────────────────────────────────────────────────────────
    const size_t  NSTEPS;       // total number of time steps
    const size_t  NX, NY;       // grid dimensions
    const size_t  n_cores;      // # of OpenMP threads (optional)
    const size_t  Z_ion;        // ionic atomic number (e.g. Z=1 for H+)
    const size_t  A_ion;        // ionic mass # (e.g. A=1 for H+)
    const double  r_ion;        // ionic radius [m] (just stored, not used here)
    const double  Lx_SI, Ly_SI; // physical domain size [m]
    const double  dt_SI;        // physical time step [s]
    const double  T_e_SI;       // electron temperature [K]
    const double  T_i_SI;       // ion temperature [K]
    const double  Ex_SI, Ey_SI; // physical external E‐field [V/m]
    const PoissonType  poisson_type; // which Poisson solver to run
    const BCType       bc_type;      // which streaming/BC we use
    const double       omega_sor;    // over‐relaxation factor for SOR

    //──────────────────────────────────────────────────────────────────────────────
    // 2) Physical Constants (SI)
    //──────────────────────────────────────────────────────────────────────────────
    static constexpr double kB_SI       = 1.380649e-23;   // [J/K]
    static constexpr double e_charge_SI = 1.602176634e-19;// [C]
    static constexpr double epsilon0_SI = 8.854187817e-12;// [F/m]
    static constexpr double m_e_SI      = 9.10938356e-31; // [kg]
    // Ion mass = A_ion * 1 u (u=1.66053906660e-27 kg)
    static constexpr double u_SI        = 1.66053906660e-27; // [kg]

    //──────────────────────────────────────────────────────────────────────────────
    // 3) Lattice‐Unit Quantities (computed once in constructor)
    //──────────────────────────────────────────────────────────────────────────────
    double dx_SI, dy_SI;       // = Lx_SI/NX, Ly_SI/NY  [m]
    double dt_latt;            // = 1.0  (in lattice units)
    double dx_latt, dy_latt;   // = 1.0  (in lattice units)
    double dt_dx;              // = dt_SI/dx_SI  (dimensionless)

    // Sound‐speeds in lattice units:   c_s^2 = (kB T / m) * (dt_SI^2 / dx_SI^2)
    double cs2_e, cs2_i, cs2_e_i, cs2_i_e;

    // Charge‐to‐mass in lattice units: (q/m) * (dt_SI^2 / dx_SI)
    double qom_e_latt, qom_i_latt;

    // Relaxation times (to be set in constructor)
    double tau_e_latt, tau_i_latt, tau_e_i_latt, tau_i_e_latt;

    // Converted E‐field in lattice units:
    //   E_latt = E_SI * (dt_SI^2 / dx_SI)
    double Ex_latt_init, Ey_latt_init;

    //──────────────────────────────────────────────────────────────────────────────
    // 4) D2Q9 Setup
    //──────────────────────────────────────────────────────────────────────────────
    static constexpr size_t   Q = 9;
    static const std::array<int, Q> cx; // = {0,1,0,-1,0,1,-1,-1,1};
    static const std::array<int, Q> cy; // = {0,0,1,0,-1,1,1,-1,-1};
    static const std::array<double, Q> w; // weights

    static const std::array<int, Q> opp;  // opposite‐direction map for bounce‐back

    //──────────────────────────────────────────────────────────────────────────────
    // 5) Per‐Node (“lattice‐unit”) Fields
    //──────────────────────────────────────────────────────────────────────────────
    // Distribution functions: f_e[i + Q*(x + NX*y)], f_i[i + Q*(x + NX*y)]
    std::vector<double>   f_e,    f_temp_e,
                         f_i,    f_temp_i;
    // Equilibrium distribution functions
    std::vector<double>   f_eq_e,    f_eq_e_i,
                         f_eq_i,    f_eq_i_e;

    // Macroscopic moments (per cell)
    std::vector<double>   n_e, n_i;      // densities
    std::vector<double>   ux_e,  uy_e,       // velocities
                         ux_i,  uy_i,
                         ux_e_i, uy_e_i;

    // Electric potential & fields (per cell), in lattice units
    std::vector<double>   phi,   phi_new;
    std::vector<double>   Ex,    Ey;         // self‐consistent E (overwrites Ex_latt_init)

    // Charge density (per cell, in SI and in lattice)
    std::vector<double>   rho_q_phys; // [C/m³]
    std::vector<double>   rho_q_latt; // dimensionless (#/cell * e_charge)

    //──────────────────────────────────────────────────────────────────────────────
    // 6) Private Methods
    //──────────────────────────────────────────────────────────────────────────────
    //Overload function to recover the index
    inline size_t INDEX(size_t x, size_t y, size_t i) const {
        return i + Q * (x + NX * y);
    }
    inline size_t INDEX(size_t x, size_t y) const {
        return x + NX * y;
    }
    //Function to reduce the lenght of program
    inline void apply_colormap(const cv::Mat& input_32f, cv::Mat& output_8uc3, int colormap) {
        static cv::Mat tmp_8u;
        cv::normalize(input_32f, tmp_8u, 0, 255, cv::NORM_MINMAX);
        tmp_8u.convertTo(tmp_8u, CV_8U);
        cv::applyColorMap(tmp_8u, output_8uc3, colormap);
    }

    // (a) Initialize all fields (set f = f_eq at t=0, zero φ, set E=Ex_latt_init)
    void Initialize();

    // (b) Compute equilibrium f_eq for given (ρ, u) and c_s^2
    void computeFeq();
    // (c) Compute forcing term (Guo) for given (u, cs2, qom_latt, Ex_cell, Ey_cell)
    void computeForceGuo(double ux, double uy,
                        double cs2,
                        double qom_latt,
                        double Ex_cell, double Ey_cell,
                        double S[Q]) const;
    //return the force
    // (d) Macroscopic update:  ρ = Σ_i f_i,   ρ u = Σ_i f_i c_i + ½ F
    void UpdateMacro();

    // (e) Collision step (BGK + forcing) for both species
    void Collisions();

    // (f) Streaming step, which calls one of:
    void Streaming();
    void Streaming_Periodic();
    void Streaming_BounceBack();

    // (g) Poisson solvers:
    void SolvePoisson();
    void SolvePoisson_GS();  // Gauss–Seidel
    void SolvePoisson_SOR(); // Successive Over‑Relaxation
    void SolvePoisson_fft();
    //add multigrid method

    // (h) Compute equilibrium distributions for both species (called inside Collisions)
    // (i) Compute new E from φ (called inside SolvePoisson)

    cv::VideoWriter video_writer;
    
};

#endif // LBMETHOD_H
