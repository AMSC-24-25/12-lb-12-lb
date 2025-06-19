#ifndef STREAMING_HPP
#define STREAMING_HPP

#include "utils.hpp"

#include <vector>
#include <array>

namespace streaming {

enum class BCType {
    Periodic,
    BounceBack
};
/// Dispatcher generico per scegliere il tipo di streaming su f e g
void Stream(std::vector<double>& fe, std::vector<double>& fi, std::vector<double>& fn, 
            std::vector<double>& temp_e, std::vector<double>& temp_i, std::vector<double>& temp_n, 
            std::vector<double>& ge, std::vector<double>& gi, std::vector<double>& gn,
            const std::array<int,Q>& cx, const std::array<int,Q>& cy, const std::array<int, Q>& opp,
            const int NX, const int NY, const BCType type);

/// Streaming massico (f) 
void StreamingPeriodic(std::vector<double>& fe, std::vector<double>& fi, std::vector<double>& fn,
                       std::vector<double>& temp_e, std::vector<double>& temp_i, std::vector<double>& temp_n, 
                       const std::array<int,Q>& cx, const std::array<int,Q>& cy,
                       const int NX, const int NY);
void StreamingBounceBack(std::vector<double>& fe, std::vector<double>& fi, std::vector<double>& fn,
                       std::vector<double>& temp_e, std::vector<double>& temp_i, std::vector<double>& temp_n, 
                       const std::array<int, Q>& cx, const std::array<int, Q>& cy, const std::array<int, Q>& opp,
                       const int NX, const int NY);

/// Streaming termico (g)
void ThermalStreamingPeriodic(std::vector<double>& ge, std::vector<double>& gi, std::vector<double>& gn,
                              std::vector<double>& temp_e, std::vector<double>& temp_i, std::vector<double>& temp_n, 
                              const std::array<int,Q>& cx, const std::array<int,Q>& cy,
                              const int NX, const int NY);
void ThermalStreamingBounceBack(std::vector<double>& ge, std::vector<double>& gi, std::vector<double>& gn,
                       std::vector<double>& temp_e, std::vector<double>& temp_i, std::vector<double>& temp_n, 
                       const std::array<int, Q>& cx, const std::array<int, Q>& cy, const std::array<int, Q>& opp,
                       const int NX, const int NY);


}  // namespace streaming

#endif  // STREAMING_HPP
