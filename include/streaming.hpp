#pragma once

#include "utils.hpp"

#include <vector>
#include <array>

namespace streaming {

enum class BCType {
    Periodic,
    BounceBack
};
//──────────────────────────────────────────────────────────────────────────────
//  Streaming dispatcher:
//──────────────────────────────────────────────────────────────────────────────
void Stream(std::vector<double>& f_e, std::vector<double>& f_i, std::vector<double>& f_n, 
            std::vector<double>& temp_e, std::vector<double>& temp_i, std::vector<double>& temp_n, 
            std::vector<double>& g_e, std::vector<double>& g_i, std::vector<double>& g_n,
            const std::array<int,Q>& cx, const std::array<int,Q>& cy,
            const int NX, const int NY, const BCType type);

//──────────────────────────────────────────────────────────────────────────────
//  Streaming + PERIODIC boundaries:
//    f_{i}(x + c_i, y + c_i, t+1) = f_{i}(x,y,t)
//──────────────────────────────────────────────────────────────────────────────
void StreamingPeriodic(std::vector<double>& f_e, std::vector<double>& f_i, std::vector<double>& f_n,
                       std::vector<double>& temp_e, std::vector<double>& temp_i, std::vector<double>& temp_n, 
                       const std::array<int,Q>& cx, const std::array<int,Q>& cy,
                       const int NX, const int NY);
//──────────────────────────────────────────────────────────────────────────────
//  Streaming + BOUNCE‐BACK walls at x=0, x=NX−1, y=0, y=NY−1.
//    If (x + c_i,x) or (y + c_i,y) is out‐of‐bounds, reflect:
//       f_{i*}(x,y) += f_{i}(x,y),
//    where i* = opp[i].
//──────────────────────────────────────────────────────────────────────────────
void StreamingBounceBack(std::vector<double>& f_e, std::vector<double>& f_i, std::vector<double>& f_n,
                       std::vector<double>& temp_e, std::vector<double>& temp_i, std::vector<double>& temp_n, 
                       const std::array<int, Q>& cx, const std::array<int, Q>& cy,
                       const int NX, const int NY);

//──────────────────────────────────────────────────────────────────────────────
//  Streaming + PERIODIC boundaries:
//    g_{i}(x + c_i, y + c_i, t+1) = g_{i}(x,y,t)l
//──────────────────────────────────────────────────────────────────────────────
void ThermalStreamingPeriodic(std::vector<double>& g_e, std::vector<double>& g_i, std::vector<double>& g_n,
                              std::vector<double>& temp_e, std::vector<double>& temp_i, std::vector<double>& temp_n, 
                              const std::array<int,Q>& cx, const std::array<int,Q>& cy,
                              const int NX, const int NY);
//──────────────────────────────────────────────────────────────────────────────
//  Streaming + BOUNCE‐BACK walls at x=0, x=NX−1, y=0, y=NY−1.
//    If (x + c_i,x) or (y + c_i,y) is out‐of‐bounds, reflect:
//       g_{i*}(x,y) += g_{i}(x,y),
//    where i* = opp[i].
//  That condition correspond to a Neumann BC so to a zero normal flux at wall
//──────────────────────────────────────────────────────────────────────────────
void ThermalStreamingBounceBack(std::vector<double>& g_e, std::vector<double>& g_i, std::vector<double>& g_n,
                       std::vector<double>& temp_e, std::vector<double>& temp_i, std::vector<double>& temp_n, 
                       const std::array<int, Q>& cx, const std::array<int, Q>& cy,
                       const int NX, const int NY);


}  // namespace streaming

