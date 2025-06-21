#pragma once

#include "utils.hpp"
#include "streaming.hpp"  // for BCType


#include <vector>
#include <cstddef>
#include <fftw3.h>
#include <cmath>


namespace poisson {

enum class PoissonType {
    NONE,
    GS,
    SOR,
    FFT,
    NPS
};

//──────────────────────────────────────────────────────────────────────────────
//  Poisson dispatcher:
//  - 5-point stencil: \nabla^2 φ = -ρ_q,  φ_new = 1/4(φ_E + φ_W + φ_N + φ_S + RHS)
//  - SOR: φ = (1-ω)φ_old + ω φ_GS
//  - 9-point stencil: φ_new = [4*(orthogonal neighbors) + (diagonals) + 6*RHS] / 20
//  - FFT: use discrete sine transform in Fourier space
//──────────────────────────────────────────────────────────────────────────────
void SolvePoisson(
    std::vector<double>& Ex,
    std::vector<double>& Ey,
    const std::vector<double>& rho_q,
    const int NX, const int NY,
    const double omega,
    const PoissonType type,
    const streaming::BCType bc_type
);


//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: Gauss–Seidel with Dirichlet φ=0 on boundary.
//    We solve  ∇² φ = − ρ_q_phys / ε₀,  in lattice units.
//    Our RHS in “lattice‐Land” is:  RHS_latt = − ρ_q_latt.
//    Then φ_new[i,j] = ¼ [ φ[i+1,j] + φ[i−1,j] + φ[i,j+1] + φ[i,j−1] − RHS_latt[i,j] ].
//──────────────────────────────────────────────────────────────────────────────
void SolvePoisson_GS(const std::vector<double>& rho_q,
                     const int NX, const int NY);
//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: GS when BC are periodic for consistency.
//──────────────────────────────────────────────────────────────────────────────
void SolvePoisson_GS_Periodic(const std::vector<double>& rho_q,
                              const int NX, const int NY);
//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: SOR (over‐relaxed Gauss–Seidel).  
//  Identical 5‐point stencil as GS, but φ_new = (1−ω) φ_old + ω φ_GS.
//──────────────────────────────────────────────────────────────────────────────
void SolvePoisson_SOR(const std::vector<double>& rho_q,
                      const int NX, const int NY, const double omega);
//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: SOR when BC are periodic for consistency.
//──────────────────────────────────────────────────────────────────────────────
void SolvePoisson_SOR_Periodic(const std::vector<double>& rho_q,
                               const int NX, const int NY, const double omega);

//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: FFT (with periodic BCs).  
//  Solves ∇²φ = –ρ_q by:
//   1) Forward real-to-complex FFT of rho_q → rho_hat(k)
//   2) Compute φ̂ (k) = –ρ̂ (k) / [4 sin²(π kx/NX) + 4 sin²(π ky/NY)]
//      (zeroing the k=0 mode to enforce zero-mean potential)
//   3) Inverse complex-to-real FFT of φ̂ → φ(x)
//   4) Normalize by NX*NY
//──────────────────────────────────────────────────────────────────────────────
void SolvePoisson_FFT(const std::vector<double>& rho_q,
                      const int NXf, const int NYf);
//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: 9-point stencil with Dirichlet φ=0 on boundary.
//    We solve  ∇²_latt φ_latt = − ρ_q_latt.
//    Then φ_new = (4*neighbors + diagonals + 6*RHS) / 20.
//
//  After convergence, we reconstruct E with centered differences:
//    E_x = −(φ[i+1,j] − φ[i−1,j]) / (2), etc.
//──────────────────────────────────────────────────────────────────────────────
void SolvePoisson_9point(const std::vector<double>& rho_q,
                         const int NX, const int NY);
//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: 9-point stencil when BC are periodic for consistency.
//──────────────────────────────────────────────────────────────────────────────
void SolvePoisson_9point_Periodic(const std::vector<double>& rho_q,
                                  const int NX, const int NY);

//──────────────────────────────────────────────────────────────────────────────
// reconstruct E = –∇φ via central differences
//──────────────────────────────────────────────────────────────────────────────
void ComputeElectricField(std::vector<double>& Ex,
                          std::vector<double>& Ey,
                          const int NX, const int NY);
//──────────────────────────────────────────────────────────────────────────────
// reconstruct E = –∇φ via central differences using Periodic BC
//──────────────────────────────────────────────────────────────────────────────
void ComputeElectricField_Periodic(std::vector<double>& Ex,
                                   std::vector<double>& Ey,
                                   const int NX, const int NY);
//──────────────────────────────────────────────────────────────────────────────
// Define for FFT objects
//──────────────────────────────────────────────────────────────────────────────
void InitPoissonFFT(const int NX, const int NY, const int NY_half, const int real_size);
//──────────────────────────────────────────────────────────────────────────────
// Destroy for FFT objects
//──────────────────────────────────────────────────────────────────────────────
void FinalizePoissonFFT();

} // namespace poisson
