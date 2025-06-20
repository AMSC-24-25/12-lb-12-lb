#include "collisions.hpp"


namespace collisions {
// Setting values for tau based on previous knowledge
const double tau_e = 5.0, tau_i = 3.0, tau_n = 1.0,
                tau_e_i = 6.0, tau_e_n = 4.0,  tau_i_n = 2.0;
//Thermal tau are considered equal to the f ones

//──────────────────────────────────────────────────────────────────────────────
// Collide function to call in the simulation
// 1) Thermal collision so that we can evealete energy loss
// 2) mass collisions
//──────────────────────────────────────────────────────────────────────────────
void Collide(
    std::vector<double>& g_e, std::vector<double>& g_i, std::vector<double>& g_n,
    const std::vector<double>& g_eq_e, const std::vector<double>& g_eq_i, const std::vector<double>& g_eq_n,
    const std::vector<double>& g_eq_e_i, const std::vector<double>& g_eq_e_n, const std::vector<double>& g_eq_i_n,
    const std::vector<double>& g_eq_i_e, const std::vector<double>& g_eq_n_e, const std::vector<double>& g_eq_n_i,
    std::vector<double>& f_e, std::vector<double>& f_i, std::vector<double>& f_n,
    const std::vector<double>& f_eq_e, const std::vector<double>& f_eq_i, const std::vector<double>& f_eq_n,
    const std::vector<double>& f_eq_e_i, const std::vector<double>& f_eq_e_n, const std::vector<double>& f_eq_i_n,
    const std::vector<double>& f_eq_i_e, const std::vector<double>& f_eq_n_e, const std::vector<double>& f_eq_n_i,
    const std::vector<double>& rho_e, const std::vector<double>& rho_i, const std::vector<double>& rho_n,
    const std::vector<double>& ux_e, const std::vector<double>& uy_e,
    const std::vector<double>& ux_i, const std::vector<double>& uy_i,
    const std::vector<double>& ux_n, const std::vector<double>& uy_n,
    const std::vector<double>& Ex, const std::vector<double>& Ey,
    const double q_e, const double q_i,
    const double m_e, const double m_i,
    std::vector<double>& temp_e, std::vector<double>& temp_i, std::vector<double>& temp_n,
    const std::array<int, Q>& cx, const std::array<int, Q>& cy, const std::array<double, Q>& w,
    const int NX, const int NY, const double Kb, const double cs2
) {
    ThermalCollisions(g_e, g_i, g_n, g_eq_e, g_eq_i, g_eq_n, 
                      g_eq_e_i, g_eq_e_n, g_eq_i_n, g_eq_i_e, g_eq_n_e, g_eq_n_i,
                      f_eq_e, f_eq_i, f_eq_n,
                      f_eq_e_i, f_eq_e_n, f_eq_i_n, f_eq_i_e, f_eq_n_e, f_eq_n_i,
                      rho_e, rho_i, rho_n,
                      ux_e, uy_e,
                      ux_i, uy_i,
                      ux_n, uy_n,
                      temp_e, temp_i, temp_n,
                      NX, NY, Kb);
    Collisions(f_e, f_i, f_n, f_eq_e, f_eq_i, f_eq_n,
                f_eq_e_i, f_eq_e_n, f_eq_i_n, f_eq_i_e, f_eq_n_e, f_eq_n_i,
                rho_e, rho_i,
                ux_e, uy_e,
                ux_i, uy_i,
                Ex, Ey,
                q_e, q_i,
                m_e, m_i,
                temp_e, temp_i, temp_n,
                cx, cy, w,
                NX, NY, cs2);
}

//──────────────────────────────────────────────────────────────────────────────
//  Thermal Collision step for both species:
//    g_e_post = g_e - (1/τ_Te)(g_e - g_e^eq) + Source
//    g_i_post = g_i - (1/τ_Ti)(g_i - g_i^eq) + Source
//  As source term we consider the enrgy losses from f
//──────────────────────────────────────────────────────────────────────────────
void ThermalCollisions(
    std::vector<double>& g_e, std::vector<double>& g_i, std::vector<double>& g_n,
    const std::vector<double>& g_eq_e, const std::vector<double>& g_eq_i, const std::vector<double>& g_eq_n,
    const std::vector<double>& g_eq_e_i, const std::vector<double>& g_eq_e_n, const std::vector<double>& g_eq_i_n,
    const std::vector<double>& g_eq_i_e, const std::vector<double>& g_eq_n_e, const std::vector<double>& g_eq_n_i,
    const std::vector<double>& f_eq_e, const std::vector<double>& f_eq_i, const std::vector<double>& f_eq_n,
    const std::vector<double>& f_eq_e_i, const std::vector<double>& f_eq_e_n, const std::vector<double>& f_eq_i_n,
    const std::vector<double>& f_eq_i_e, const std::vector<double>& f_eq_n_e, const std::vector<double>& f_eq_n_i,
    const std::vector<double>& rho_e, const std::vector<double>& rho_i, const std::vector<double>& rho_n,
    const std::vector<double>& ux_e, const std::vector<double>& uy_e,
    const std::vector<double>& ux_i, const std::vector<double>& uy_i,
    const std::vector<double>& ux_n, const std::vector<double>& uy_n,
    std::vector<double>& temp_e, std::vector<double>& temp_i, std::vector<double>& temp_n, 
    const int NX, const int NY, const double Kb
) {
    #pragma omp parallel for collapse(3)
    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            for (int i = 0; i < Q; ++i) {
                const int idx_3 = INDEX(x, y, i,NX,Q);
                const int idx_2 = INDEX(x, y,NX);

                const double term_ee=(2.0*rho_e[idx_2]*(1.0-1.0/tau_e)*(1.0-1.0/tau_e)-2.0*(1.0-1.0/tau_e)*rho_e[idx_2]-Q*f_eq_e[idx_3]/tau_e)/(2.0*(2.0*(1.0-1.0/tau_e)+Q*f_eq_e[idx_3]/tau_e));
                const double term_ei=(2.0*rho_e[idx_2]*(1.0-1.0/tau_e_i)*(1.0-1.0/tau_e_i)-2.0*(1.0-1.0/tau_e_i)*rho_e[idx_2]-Q*f_eq_e_i[idx_3]/tau_e_i)/(2.0*(2.0*(1.0-1.0/tau_e_i)+Q*f_eq_e_i[idx_3]/tau_e_i));
                const double term_en=(2.0*rho_e[idx_2]*(1.0-1.0/tau_e_n)*(1.0-1.0/tau_e_n)-2.0*(1.0-1.0/tau_e_n)*rho_e[idx_2]-Q*f_eq_e_n[idx_3]/tau_e_n)/(2.0*(2.0*(1.0-1.0/tau_e_n)+Q*f_eq_e_n[idx_3]/tau_e_n));

                const double term_ii=(2.0*rho_i[idx_2]*(1.0-1.0/tau_i)*(1.0-1.0/tau_i)-2.0*(1.0-1.0/tau_i)*rho_i[idx_2]-Q*f_eq_i[idx_3]/tau_i)/(2.0*(2.0*(1.0-1.0/tau_i)+Q*f_eq_i[idx_3]/tau_i));
                const double term_ie=(2.0*rho_i[idx_2]*(1.0-1.0/tau_e_i)*(1.0-1.0/tau_e_i)-2.0*(1.0-1.0/tau_e_i)*rho_i[idx_2]-Q*f_eq_i_e[idx_3]/tau_e_i)/(2.0*(2.0*(1.0-1.0/tau_e_i)+Q*f_eq_i_e[idx_3]/tau_e_i));
                const double term_in=(2.0*rho_i[idx_2]*(1.0-1.0/tau_i_n)*(1.0-1.0/tau_i_n)-2.0*(1.0-1.0/tau_i_n)*rho_i[idx_2]-Q*f_eq_i_n[idx_3]/tau_i_n)/(2.0*(2.0*(1.0-1.0/tau_i_n)+Q*f_eq_i_n[idx_3]/tau_i_n));

                const double term_nn=(2.0*rho_n[idx_2]*(1.0-1.0/tau_n)*(1.0-1.0/tau_n)-2.0*(1.0-1.0/tau_n)*rho_n[idx_2]-Q*f_eq_n[idx_3]/tau_n)/(2.0*(2.0*(1.0-1.0/tau_n)+Q*f_eq_n[idx_3]/tau_n));
                const double term_ne=(2.0*rho_n[idx_2]*(1.0-1.0/tau_e_n)*(1.0-1.0/tau_e_n)-2.0*(1.0-1.0/tau_e_n)*rho_n[idx_2]-Q*f_eq_n_e[idx_3]/tau_e_n)/(2.0*(2.0*(1.0-1.0/tau_e_n)+Q*f_eq_n_e[idx_3]/tau_e_n));
                const double term_ni=(2.0*rho_n[idx_2]*(1.0-1.0/tau_i_n)*(1.0-1.0/tau_i_n)-2.0*(1.0-1.0/tau_i_n)*rho_n[idx_2]-Q*f_eq_n_i[idx_3]/tau_i_n)/(2.0*(2.0*(1.0-1.0/tau_i_n)+Q*f_eq_n_i[idx_3]/tau_i_n));

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

                // Update distribution functions with Source
                temp_e[idx_3] = g_e[idx_3]+ C_Te + DeltaT_e;
                temp_i[idx_3] = g_i[idx_3]+ C_Ti + DeltaT_i;
                temp_n[idx_3] = g_n[idx_3]+ C_Tn + DeltaT_n;
            }
        }
    }
    // Swap temporary arrays with main arrays
    g_e.swap(temp_e);
    g_i.swap(temp_i);
    g_n.swap(temp_n);
}
//──────────────────────────────────────────────────────────────────────────────
//  Collision step (BGK + Guo forcing) for both species:
//    f_e_post = f_e - (1/τ_e)(f_e - f_e^eq) + F_e
//    f_i_post = f_i - (1/τ_i)(f_i - f_i^eq) + F_i
//──────────────────────────────────────────────────────────────────────────────
void Collisions(
    std::vector<double>& f_e, std::vector<double>& f_i, std::vector<double>& f_n,
    const std::vector<double>& f_eq_e, const std::vector<double>& f_eq_i, const std::vector<double>& f_eq_n,
    const std::vector<double>& f_eq_e_i, const std::vector<double>& f_eq_e_n, const std::vector<double>& f_eq_i_n,
    const std::vector<double>& f_eq_i_e, const std::vector<double>& f_eq_n_e, const std::vector<double>& f_eq_n_i,
    const std::vector<double>& rho_e, const std::vector<double>& rho_i,
    const std::vector<double>& ux_e, const std::vector<double>& uy_e,
    const std::vector<double>& ux_i, const std::vector<double>& uy_i,
    const std::vector<double>& Ex, const std::vector<double>& Ey,
    const double q_e, const double q_i,
    const double m_e, const double m_i,
    std::vector<double>& temp_e, std::vector<double>& temp_i, std::vector<double>& temp_n,
    const std::array<int, Q>& cx, const std::array<int, Q>& cy, const std::array<double, Q>& w,
    const int NX, const int NY, const double cs2
) {
    #pragma omp parallel for collapse(2)
    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            const int idx = INDEX(x, y,NX);
            
            const double Ex_loc = Ex[idx];
            const double Ey_loc = Ey[idx];

            for (int i = 0; i < Q; ++i) {
                const int idx_3 = INDEX(x, y, i,NX,Q);
                
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
                temp_e[idx_3] = f_e[idx_3] + C_e + F_e;
                temp_i[idx_3] = f_i[idx_3] + C_i + F_i;
                temp_n[idx_3] = f_n[idx_3] + C_n;
            }
        }
    }
    // Swap temporary arrays with main arrays
    f_e.swap(temp_e);
    f_i.swap(temp_i);
    f_n.swap(temp_n);
}


} // namespace collisions
