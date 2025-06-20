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
#include <omp.h>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <sstream>

//--------------------------------------------------------------------------------
// Enumerations for choosing Poisson solver and Streaming/BC type
//--------------------------------------------------------------------------------
enum class PoissonType { NONE = 0, GAUSS_SEIDEL = 1, SOR = 2, FFT = 3, NPS =4};
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
             const double    Ex_SI,
             const double    Ey_SI,
             const double    T_e_SI_init,
             const double    T_i_SI_init,
             const double    T_n_SI_init,
             const double    n_e_SI_init,
             const double    n_n_SI_init,
             const PoissonType poisson_type,
             const BCType      bc_type,
             const double    omega_sor);

    // Run the complete simulation (calls Initialize(), then loops on TimeSteps)
    void Run_simulation();

    // File unico di debug
    std::ofstream debug_file;
    // Inizializza il file di debug (da chiamare una volta prima del loop)
    void InitDebugDump(const std::string& filename="debug_dump.txt");
    // Scarica lo stato dopo un certo step e una certa fase ("UpdateMacro", "Collisions", ...)
    void DumpGridStateReadable(size_t step, const std::string& stage);
    // Chiude il file di debug (da chiamare alla fine)
    void CloseDebugDump();

private:
    //──────────────────────────────────────────────────────────────────────────────
    // 1) “Raw” (SI) Inputs
    //──────────────────────────────────────────────────────────────────────────────
    const size_t  NSTEPS;       // total number of time steps
    const size_t  NX, NY;       // grid dimensions
    const size_t  n_cores;      // # of OpenMP threads (optional)
    const size_t  Z_ion;        // ionic atomic number (e.g. Z=1 for H+)
    const size_t  A_ion;        // ionic mass # (e.g. A=1 for H+)
    const double    Ex_SI;
    const double    Ey_SI;
    const double    T_e_SI_init;
    const double    T_i_SI_init;
    const double    T_n_SI_init;
    const double    n_e_SI_init;
    const double    n_n_SI_init;
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
    static constexpr double u_SI        = 1.66053906660e-27; // [kg]
    static constexpr double m_p_SI      = 1.67262192595e-27; // [kg]
    static constexpr double m_ne_SI      = 1.67492749804e-27; // [kg]

    const double m_i_SI = A_ion * u_SI; //[kg]
    const double m_n_SI = A_ion * u_SI; //[kg]
    //──────────────────────────────────────────────────────────────────────────────
    // Physical conversion quantities from SI to LU:
    //──────────────────────────────────────────────────────────────────────────────
    const double n0_SI = n_e_SI_init;

    const double M0_SI = m_e_SI; // physical mass [kg]
    const double T0_SI = T_e_SI_init; // physical temperature [K]
    const double Q0_SI = e_charge_SI; // physical charge [C]
    const double L0_SI = std::sqrt(epsilon0_SI * kB_SI * T0_SI / (n0_SI * Q0_SI * Q0_SI))*1e-2; // physical lenght = lambda_D [m]
    const double t0_SI = std::sqrt(epsilon0_SI * M0_SI / (3.0 * n0_SI * Q0_SI * Q0_SI))  *1e-2; // physical time = rad(3)/w_p [s]
    //other useful obtained scaling quantities
    const double E0_SI = M0_SI*L0_SI/(Q0_SI*t0_SI*t0_SI); // physical electric field [V/m]
    const double v0_SI = L0_SI / t0_SI; // physical velocity [m/s]
    const double F0_SI = M0_SI * L0_SI / (t0_SI * t0_SI); // physical force [N]

    //──────────────────────────────────────────────────────────────────────────────
    // 3) Lattice‐Unit Quantities rescaled here
    //──────────────────────────────────────────────────────────────────────────────
    // Sound‐speeds in lattice units from D2Q9 c_s^2=1/3
    const double cs2 = kB_SI * T0_SI / M0_SI * t0_SI * t0_SI / (L0_SI * L0_SI);
    //If we define the kimetic viscosity we are able to retrive from that the values for tau
    //const double nu_e=7.1e10, nu_i=1.8e5, nu_n=8.3e3, 
    //             nu_e_i=3.6e10, nu_e_n=2.1e6, nu_i_n=8.3e2;
    //const double tau_e=nu_e*t0_SI/(cs2*L0_SI*L0_SI)+0.5, tau_i=nu_i*t0_SI/(cs2*L0_SI*L0_SI)+0.5, tau_n=nu_n*t0_SI/(cs2*L0_SI*L0_SI)+0.5, 
    //             tau_e_i=nu_e_i*t0_SI/(cs2*L0_SI*L0_SI)+0.5, tau_e_n=nu_e*t0_SI/(cs2*L0_SI*L0_SI)+0.5, tau_i_n=nu_e*t0_SI/(cs2*L0_SI*L0_SI)+0.5;
   
    const double Kb = kB_SI* (t0_SI * t0_SI * T0_SI)/(L0_SI * L0_SI * M0_SI);

    // Otherwise we have to set values for tau based on previous knowledge
    const double tau_e = 5.0, tau_i = 3.0, tau_n = 1.0,
                 tau_e_i = 6.0, tau_e_n = 4.0,  tau_i_n = 2.0;
    //Thermal tau are considered equal to the f ones
    
    // Converted E‐field in lattice units:
    const double Ex_ext = Ex_SI / E0_SI, 
                 Ey_ext = Ey_SI / E0_SI; // external E‐field in lattice units

    // Converted temperatures in lattice units:
    const double T_e_init = T_e_SI_init / T0_SI, 
                 T_i_init = T_i_SI_init / T0_SI,
                 T_n_init = T_n_SI_init / T0_SI; // initial temperatures in lattice units

    // mass in lattice units:
    const double m_e = m_e_SI / M0_SI, // electron mass in lattice units
                 m_i = m_i_SI / M0_SI, // ion mass in electron masses (for convenience)
                 m_n = m_n_SI / M0_SI;

    // Converted charge in lattice units:
    const double q_e = - e_charge_SI / Q0_SI; // electron charge in lattice units
    const double q_i = Z_ion * e_charge_SI / Q0_SI; // ion charge in lattice units

    // Initial density in lattice unit
    const double rho_e_init = m_e * n_e_SI_init / n0_SI, // electron density in lattice units
                 rho_i_init = m_i * n_e_SI_init / n0_SI / Z_ion, // ion density in lattice units. The idea behind /Z_ion is the quasi neutrality of the plamsa at the start
                 rho_n_init = m_n * n_n_SI_init / n0_SI;
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
                          f_i,    f_temp_i,
                          f_n,    f_temp_n;
    // Equilibrium distribution functions
    std::vector<double>   f_eq_e,    f_eq_i,    f_eq_n,   
                          f_eq_e_i,  f_eq_i_e,
                          f_eq_e_n,  f_eq_n_e,
                          f_eq_i_n,  f_eq_n_i;
                         
    // Thermal distribution function
    std::vector<double>   g_e,    g_temp_e,
                          g_i,    g_temp_i,
                          g_n,    g_temp_n;
    // Equilibrium distribution functions
    std::vector<double>   g_eq_e,    g_eq_i,    g_eq_n,
                          g_eq_e_i,  g_eq_i_e,
                          g_eq_e_n,  g_eq_n_e,
                          g_eq_i_n,  g_eq_n_i;

    // Macroscopic moments (per cell)
    std::vector<double>   rho_e,  rho_i, rho_n;      // densities
    std::vector<double>   ux_e,   uy_e,       // velocities
                          ux_i,   uy_i,
                          ux_n,   uy_n,
                          ux_e_i, uy_e_i,
                          ux_e_n, uy_e_n,
                          ux_i_n, uy_i_n;
    
    // Temperature vectors
    std::vector<double>  T_e,  T_i, T_n;

    // Electric potential & fields (per cell), in lattice units
    std::vector<double>   phi,   phi_new;
    std::vector<double>   Ex,    Ey;         // self‐consistent E (overwrites Ex_latt_init)

    // Charge density (per cell in lattice units)
    std::vector<double>   rho_q; // dimensionless (#/cell * e_charge)

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


    // Helper: create legend panel (JET colormap)
    cv::Mat makeColorLegend(double min_val, double max_val, int height, int width = 40, int text_area = 60, int border = 10) {
        cv::Mat gray(height, 1, CV_8U);
        for (int i = 0; i < height; ++i)
            gray.at<uchar>(i, 0) = 255 - (i * 255 / (height - 1));
        cv::Mat colorbar;
        cv::applyColorMap(gray, colorbar, cv::COLORMAP_JET);
        cv::resize(colorbar, colorbar, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);

        cv::Mat panel(height + 2 * border, width + text_area, CV_8UC3, cv::Scalar(255, 255, 255));
        colorbar.copyTo(panel(cv::Rect(border, border, width, height)));

        auto putVal = [&](const std::string &txt, int y) {
            cv::putText(panel, txt,
                        cv::Point(border + width + 5, y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 0), 1);
        };

        std::ostringstream oss_max, oss_mid, oss_min;
        oss_max << std::fixed << std::setprecision(2) << max_val;
        oss_mid << std::fixed << std::setprecision(2) << 0.5 * (min_val + max_val);
        oss_min << std::fixed << std::setprecision(2) << min_val;

        putVal(oss_max.str(), border + 10);
        putVal(oss_mid.str(), border + height / 2);
        putVal(oss_min.str(), border + height - 5);

        return panel;
    }

    // (a) Initialize all fields (set f = f_eq at t=0, zero φ, set E=Ex_latt_init)
    void Initialize();

    // (b) Compute equilibrium f_eq for given (ρ, u) and c_s^2
    void ComputeEquilibrium();
    
    // (d) Macroscopic update:  ρ = Σ_i f_i,   ρ u = Σ_i f_i c_i + ½ F
    void UpdateMacro();

    // (e) Collision step (BGK + forcing) for both species
    void Collisions();
    
    void ThermalCollisions();

    // (f) Streaming step, which calls one of:
    void Streaming();
    void Streaming_Periodic();
    void Streaming_BounceBack();

    void ThermalStreaming_Periodic();
    void ThermalStreaming_BounceBack();

    // (g) Poisson solvers:
    void SolvePoisson();
    void SolvePoisson_GS();  // Gauss–Seidel
    void SolvePoisson_GS_Periodic();
    void SolvePoisson_SOR(); // Successive Over‑Relaxation
    void SolvePoisson_SOR_Periodic();
    void SolvePoisson_fft();
    void SolvePoisson_9point();
    void SolvePoisson_9point_Periodic();

    // Visualization function to see the movement in OpenCV.
    void VisualizationDensity();
    void VisualizationVelocity();
    void VisualizationTemperature();

    // (h) Compute equilibrium distributions for both species (called inside Collisions)
    // (i) Compute new E from φ (called inside SolvePoisson)
    
    // Functions to plot grahps of values
    void InitTimeSeries();
    void RecordTimeSeriesStep(size_t t);
    void FinalizeTimeSeriesPlots();

    // the 9 sample points
    std::vector<std::pair<size_t,size_t>> sample_points;

    // a shared time‐axis
    std::vector<double> ts;

    // for each quantity, a history[time][point]
    std::vector<std::vector<double>> hist_ux_e,
                                   hist_uy_e,
                                   hist_ue_mag,
                                   hist_ux_i,
                                   hist_uy_i,
                                   hist_ui_mag,
                                   hist_ux_n,
                                   hist_uy_n,
                                   hist_un_mag,
                                   hist_T_e,
                                   hist_T_i,
                                   hist_T_n,
                                   hist_rho_e,
                                   hist_rho_i,
                                   hist_rho_n,
                                   hist_rho_q,
                                   hist_Ex,
                                   hist_Ey,
                                   hist_E_mag;

    // helper to plot one set of histories
    void PlotTimeSeriesDirect(const std::string &png_filename,
                            const std::string &title,
                            const std::vector<std::string> &legends,
                            const std::vector<std::vector<double>> &data);

    cv::VideoWriter video_writer_density, video_writer_velocity, video_writer_temperature;
    // Global‐range trackers for visualization:
    // --- Density and charge visualization ranges
    static constexpr double DENSITY_E_MIN = 0.0;
    static constexpr double DENSITY_E_MAX = 1.0;
    static constexpr double DENSITY_I_MIN = 0.0;
    static constexpr double DENSITY_I_MAX = 1822.0;
    static constexpr double CHARGE_MIN  = 0.5;
    static constexpr double CHARGE_MAX  = 1.5;

    // --- Velocity visualization ranges
    static constexpr double UX_E_MIN = -1e-7;
    static constexpr double UX_E_MAX =  1e-7;
    static constexpr double UY_E_MIN = -1e-7;
    static constexpr double UY_E_MAX =  1e-7;
    static constexpr double UE_MAG_MIN = 0.0;
    static constexpr double UE_MAG_MAX = 1e-7;

    static constexpr double UX_I_MIN = -1e-7;
    static constexpr double UX_I_MAX =  1e-7;
    static constexpr double UY_I_MIN = -1e-7;
    static constexpr double UY_I_MAX =  1e-7;
    static constexpr double UI_MAG_MIN = 0.0;
    static constexpr double UI_MAG_MAX = 1e-7;

    // --- Temperature visualization ranges
    static constexpr double TEMP_E_MIN = 1e-6;
    static constexpr double TEMP_E_MAX = 1;
    static constexpr double TEMP_I_MIN = 1e-6;
    static constexpr double TEMP_I_MAX = 0.5;

    
};

#endif // LBMETHOD_H
