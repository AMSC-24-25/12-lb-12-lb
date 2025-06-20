#pragma once

#include "utils.hpp"

#include <vector>
#include <array>

namespace collisions {

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
);
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
);
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
);


} // namespace collisions
