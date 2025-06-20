#include "poisson.hpp"

#include <vector>
#include <mutex> // for call_once
#include <algorithm>  // for std::fill

namespace poisson {

static std::vector<double> phi;
static std::once_flag poisson_init_flag, fftw_init_flag;

// Variables for GS, SOR, 9ps convergence
static constexpr size_t maxIter = 5000;
static constexpr double tol = 1e-8;
//if not used (FFT) compiler eliminates it automatically

// Variables for FFT
static double* fft_in = nullptr;
static double* fft_out = nullptr;
static fftw_complex* fft_rho_hat = nullptr;
static fftw_complex* fft_phi_hat = nullptr;
static fftw_plan fft_plan_r2c;
static fftw_plan fft_plan_c2r;

void SolvePoisson(
    std::vector<double>& Ex,
    std::vector<double>& Ey,
    const std::vector<double>& rho_q,
    const int NX, const int NY,
    const double omega,
    const PoissonType type,
    const streaming::BCType bc_type
) {
    std::call_once(poisson_init_flag, [&]() {
        phi.resize(NX * NY, 0.0);
        if(type==PoissonType::NONE){
            std::fill(Ex.begin(), Ex.end(), 0.0);
            std::fill(Ey.begin(), Ey.end(), 0.0);
            return;
        }
    });
    // If Poisson not solved return
    if (type == PoissonType::NONE) return; //We prefer it here because the compiler optimize it better
                                           //An alternative position is inside the swithc as default
    // Dispatcher based on Poisson Solver and BC
    if(bc_type==streaming::BCType::Periodic){
        switch(type){
            case PoissonType::GS:
                SolvePoisson_GS(rho_q, NX, NY);
                break;
            case PoissonType::SOR:
                SolvePoisson_SOR(rho_q, NX, NY, omega);
                break;
            case PoissonType::NPS:
                SolvePoisson_9point(rho_q, NX, NY);
                break;
            case PoissonType::FFT:
                SolvePoisson_FFT(rho_q, NX, NY);
                break;
            default:
                return;
        }
        ComputeElectricField_Periodic(Ex, Ey, NX, NY);
    }
    else{
        switch(type){
            case PoissonType::GS:
                SolvePoisson_GS(rho_q, NX, NY);
                break;
            case PoissonType::SOR:
                SolvePoisson_SOR(rho_q, NX, NY, omega);
                break;
            case PoissonType::NPS:
                SolvePoisson_9point(rho_q, NX, NY);
                break;
            default://If we are defining FFT with non periodic simply exit without solving Poisson
                return;
        }
        ComputeElectricField(Ex, Ey, NX, NY);
    }
  
}

//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: Gauss–Seidel with Dirichlet φ=0 on boundary.
//    We solve  ∇² φ = − ρ_q_phys / ε₀,  in lattice units.
//    Our RHS in “lattice‐Land” is:  RHS_latt = − ρ_q_latt.
//    Then φ_new[i,j] = ¼ [ φ[i+1,j] + φ[i−1,j] + φ[i,j+1] + φ[i,j−1] − RHS_latt[i,j] ].
//──────────────────────────────────────────────────────────────────────────────
void SolvePoisson_GS(const std::vector<double>& rho_q,
                      const int NX, const int NY){

    // Red–Black Gauss–Seidel iteration for parallelism:
    // we split the grid into two interleaved sets ("red" and "black")
    // so each set can be updated in parallel without data races.
    for (size_t iter = 0; iter < maxIter; ++iter) {
        double maxErr = 0.0;

        // Update "red" points: (i+j) even
        #pragma omp parallel for collapse(2) reduction(max:maxErr)
        for (int j = 1; j < NY - 1; ++j) {
            for (int i = 1; i < NX - 1; ++i) {
                if (((i + j) & 1) == 0) {
                    const int idx = INDEX(i, j, NX);
                    // sum of four neighbors
                    const double nb = phi[INDEX(i+1, j, NX)]
                              + phi[INDEX(i-1, j, NX)]
                              + phi[INDEX(i, j+1, NX)]
                              + phi[INDEX(i, j-1, NX)];
                    const double newPhi = 0.25 * (nb + rho_q[idx]);
                    const double err    = fabs(newPhi - phi[idx]);
                    phi[idx] = newPhi;
                    if (err > maxErr) maxErr = err;
                }
            }
        }

        // Update "black" points: (i+j) odd
        #pragma omp parallel for collapse(2) reduction(max:maxErr)
        for (int j = 1; j < NY - 1; ++j) {
            for (int i = 1; i < NX - 1; ++i) {
                if (((i + j) & 1) == 1) {
                    const int idx = INDEX(i, j,NX);
                    const double nb = phi[INDEX(i+1, j,NX)]
                              + phi[INDEX(i-1, j,NX)]
                              + phi[INDEX(i, j+1,NX)]
                              + phi[INDEX(i, j-1,NX)];
                    const double newPhi = 0.25 * (nb + rho_q[idx]);
                    const double err    = fabs(newPhi - phi[idx]);
                    phi[idx] = newPhi;
                    if (err > maxErr) maxErr = err;
                }
            }
        }

        // Check for convergence
        if (maxErr < tol) {
            // early exit if solution has converged
            break;
        }
    }
}
//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: GS when BC are periodic for consistency.
//──────────────────────────────────────────────────────────────────────────────
void SolvePoisson_GS_Periodic(const std::vector<double>& rho_q,
                              const int NX, const int NY){

    for (size_t iter = 0; iter < maxIter; ++iter) {
        double maxErr = 0.0;

        // === RED sweep (i+j even) ===
        #pragma omp parallel for collapse(2) reduction(max:maxErr)
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                if (((i + j) & 1) == 0) {
                    // periodic neighbors
                    const int ip = (i + 1) % NX;
                    const int im = (i + NX - 1) % NX;
                    const int jp = (j + 1) % NY;
                    const int jm = (j + NY - 1) % NY;

                    const int idx = INDEX(i, j,NX);
                    const double sumNb =
                        phi[INDEX(ip, j,NX)] +
                        phi[INDEX(im, j,NX)] +
                        phi[INDEX(i, jp,NX)] +
                        phi[INDEX(i, jm,NX)];

                    const double newPhi = 0.25 * (sumNb + rho_q[idx]);
                    const double err    = fabs(newPhi - phi[idx]);
                    phi[idx]      = newPhi;

                    if (err > maxErr) maxErr = err;
                }
            }
        }

        // === BLACK sweep (i+j odd) ===
        #pragma omp parallel for collapse(2) reduction(max:maxErr)
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                if (((i + j) & 1) == 1) {
                    // periodic neighbors
                    const int ip = (i + 1) % NX;
                    const int im = (i + NX - 1) % NX;
                    const int jp = (j + 1) % NY;
                    const int jm = (j + NY - 1) % NY;

                    const int idx = INDEX(i, j,NX);
                    const double sumNb =
                        phi[INDEX(ip, j,NX)] +
                        phi[INDEX(im, j,NX)] +
                        phi[INDEX(i, jp,NX)] +
                        phi[INDEX(i, jm,NX)];

                    const double newPhi = 0.25 * (sumNb + rho_q[idx]);
                    const double err    = fabs(newPhi - phi[idx]);
                    phi[idx]      = newPhi;

                    if (err > maxErr) maxErr = err;
                }
            }
        }

        // Convergence check
        if (maxErr < tol) {
            break;
        }
    }
}
//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: SOR (over‐relaxed Gauss–Seidel).  
//  Identical 5‐point stencil as GS, but φ_new = (1−ω) φ_old + ω φ_GS.
//──────────────────────────────────────────────────────────────────────────────
void SolvePoisson_SOR(const std::vector<double>& rho_q,
                      const int NX, const int NY, const double omega){

    for (size_t iter = 0; iter < maxIter; ++iter) {
        double maxErr = 0.0;

        // === RED sweep (i+j even) ===
        #pragma omp parallel for collapse(2) reduction(max:maxErr)
        for (int j = 1; j < NY - 1; ++j) {
            for (int i = 1; i < NX - 1; ++i) {
                if (((i + j) & 1) == 0) {
                    const int idx = INDEX(i, j,NX);
                    const double oldPhi = phi[idx];

                    // sum of neighbors (Gauss–Seidel stencil)
                    const double nb = phi[INDEX(i+1, j,NX)]
                              + phi[INDEX(i-1, j,NX)]
                              + phi[INDEX(i, j+1,NX)]
                              + phi[INDEX(i, j-1,NX)];

                    // standard Gauss–Seidel update
                    const double gsPhi = 0.25 * (nb + rho_q[idx]);

                    // SOR update: blend old and GS value
                    const double newPhi = (1.0 - omega) * oldPhi
                                  + omega       * gsPhi;

                    phi[idx] = newPhi;
                    const double err = fabs(newPhi - oldPhi);
                    if (err > maxErr) maxErr = err;
                }
            }
        }

        // === BLACK sweep (i+j odd) ===
        #pragma omp parallel for collapse(2) reduction(max:maxErr)
        for (int j = 1; j < NY - 1; ++j) {
            for (int i = 1; i < NX - 1; ++i) {
                if (((i + j) & 1) == 1) {
                    const int idx = INDEX(i, j,NX);
                    const double oldPhi = phi[idx];

                    const double nb = phi[INDEX(i+1, j,NX)]
                              + phi[INDEX(i-1, j,NX)]
                              + phi[INDEX(i, j+1,NX)]
                              + phi[INDEX(i, j-1,NX)];

                    const double gsPhi = 0.25 * (nb + rho_q[idx]);
                    const double newPhi = (1.0 - omega) * oldPhi
                                  + omega       * gsPhi;

                    phi[idx] = newPhi;
                    const double err = fabs(newPhi - oldPhi);
                    if (err > maxErr) maxErr = err;
                }
            }
        }

        // check convergence
        if (maxErr < tol) {
            break;  // solution converged early
        }
    }
}
//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: SOR when BC are periodic for consistency.
//──────────────────────────────────────────────────────────────────────────────
void SolvePoisson_SOR_Periodic(const std::vector<double>& rho_q,
                               const int NX, const int NY, const double omega){

    for (size_t iter = 0; iter < maxIter; ++iter) {
        double maxErr = 0.0;

        // === RED sweep (i+j even) ===
        #pragma omp parallel for collapse(2) reduction(max:maxErr)
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                if (((i + j) & 1) == 0) {
                    // periodic neighbor indices
                    const int ip = (i + 1) % NX;
                    const int im = (i + NX - 1) % NX;
                    const int jp = (j + 1) % NY;
                    const int jm = (j + NY - 1) % NY;

                    const int idx = INDEX(i, j,NX);
                    const double oldPhi = phi[idx];

                    // Gauss–Seidel stencil sum
                    const double sumNb = phi[INDEX(ip, j,NX)]
                                 + phi[INDEX(im, j,NX)]
                                 + phi[INDEX(i, jp,NX)]
                                 + phi[INDEX(i, jm,NX)];

                    const double gsPhi   = 0.25 * (sumNb + rho_q[idx]);
                    const double newPhi  = (1.0 - omega) * oldPhi
                                   + omega       * gsPhi;
                    phi[idx]       = newPhi;

                    const double err = fabs(newPhi - oldPhi);
                    if (err > maxErr) maxErr = err;
                }
            }
        }

        // === BLACK sweep (i+j odd) ===
        #pragma omp parallel for collapse(2) reduction(max:maxErr)
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                if (((i + j) & 1) == 1) {
                    const int ip = (i + 1) % NX;
                    const int im = (i + NX - 1) % NX;
                    const int jp = (j + 1) % NY;
                    const int jm = (j + NY - 1) % NY;

                    const int idx = INDEX(i, j,NX);
                    const double oldPhi = phi[idx];

                    const double sumNb = phi[INDEX(ip, j,NX)]
                                 + phi[INDEX(im, j,NX)]
                                 + phi[INDEX(i, jp,NX)]
                                 + phi[INDEX(i, jm,NX)];

                    const double gsPhi   = 0.25 * (sumNb + rho_q[idx]);
                    const double newPhi  = (1.0 - omega) * oldPhi
                                   + omega       * gsPhi;
                    phi[idx]       = newPhi;

                    const double err = fabs(newPhi - oldPhi);
                    if (err > maxErr) maxErr = err;
                }
            }
        }

        // convergence check
        if (maxErr < tol) {
            break;
        }
    }
}

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
                      const int NX, const int NY){
    // rho_q[i] is charge density in lattice units.
    // The net charge should be approximately zero.
    // If not, the k=0 component will be removed → potential average = 0.
    // Useful variables for the method
    const int NY_half = NY / 2 + 1; //only NYf/2 +1 are unique coefficients
    const int real_size = NX * NY;  //Domain in the real space
    std::call_once(fftw_init_flag, [&]() {
        InitPoissonFFT(NX, NY, NY_half, real_size);
        std::atexit(FinalizePoissonFFT);  // with that we weill free the memory at the end of the code
    });
    // Copy input charge density into FFT input array
    #pragma omp parallel for
    for (int idx = 0; idx < real_size; ++idx) {
        fft_in[idx] = rho_q[idx];
    }
    
    // Forward FFT: rho_q -> rho_hat
    fftw_execute(fft_plan_r2c);

    // Solve Poisson's equation in Fourier space:
    // ∇²φ = -rho → φ_hat = rho_hat / (kx² + ky²)
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY_half; ++j) {
            const int kx = (i <= NX / 2) ? i : i - NX; //this ensures that kx index are centered around 0
            const int ky = j;
            const double sinx = std::sin(M_PI * kx / NX);
            const double siny = std::sin(M_PI * ky / NY);
            const double denom = 4.0 * (sinx * sinx + siny * siny);

            const int index = i * NY_half + j;

            if (denom > 1e-15) {
                fft_phi_hat[index][0] = fft_rho_hat[index][0] / denom;
                fft_phi_hat[index][1] = fft_rho_hat[index][1] / denom;
            } else {
                // Set zero mode to zero to enforce zero average potential (Gauge condition)
                // This is done for the zeroth mode φ_hat(0)
                fft_phi_hat[index][0] = 0.0;
                fft_phi_hat[index][1] = 0.0;
            }
        }
    }

    // Inverse FFT: phi_hat -> out
    fftw_execute(fft_plan_c2r);// This is not yet normalized so we need to normalize it

    // Normalize result (FFTW does not normalize the inverse transform)
    const double norm = 1.0 / real_size; //Implicit conversion no need for casting
    #pragma omp parallel for
    for (int idx = 0; idx < real_size; ++idx) {
        phi[idx] = fft_out[idx] * norm;
    }
}
//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: 9-point stencil with Dirichlet φ=0 on boundary.
//    We solve  ∇²_latt φ_latt = − ρ_q_latt.
//    Then φ_new = (4*neighbors + diagonals + 6*RHS) / 20.
//
//  After convergence, we reconstruct E with centered differences:
//    E_x = −(φ[i+1,j] − φ[i−1,j]) / (2), etc.
//──────────────────────────────────────────────────────────────────────────────
void SolvePoisson_9point(const std::vector<double>& rho_q,
                         const int NX, const int NY){

    // Iterate until convergence or maxIter
    for (size_t iter = 0; iter < maxIter; ++iter) {
        double maxErr = 0.0;

        // We need 4 colors so that no two stencil neighbors share the same color.
        // color = 2*(i%2) + (j%2) ∈ {0,1,2,3}
        for (std::uint8_t sweep = 0; sweep < 4; ++sweep) { //using unit8 for less memory occupation
            double maxErrSweep = 0.0;

            #pragma omp parallel for collapse(2) reduction(max:maxErrSweep)
            for (int j = 1; j < NY - 1; ++j) {
                for (int i = 1; i < NX - 1; ++i) {
                    // determine checkerboard color
                    if ((2 * (i & 1) + (j & 1)) != sweep) 
                        continue;

                    const int idx       = INDEX(i, j,NX);
                    const int ip1_j     = INDEX(i+1, j,NX);
                    const int im1_j     = INDEX(i-1, j,NX);
                    const int i_jp1     = INDEX(i, j+1,NX);
                    const int i_jm1     = INDEX(i, j-1,NX);
                    const int ip1_jp1   = INDEX(i+1, j+1,NX);
                    const int im1_jp1   = INDEX(i-1, j+1,NX);
                    const int ip1_jm1   = INDEX(i+1, j-1,NX);
                    const int im1_jm1   = INDEX(i-1, j-1,NX);

                    // orthogonal neighbors
                    const double sumOrtho = phi[ip1_j] + phi[im1_j]
                                    + phi[i_jp1] + phi[i_jm1];
                    // diagonal neighbors
                    const double sumDiag  = phi[ip1_jp1] + phi[im1_jp1]
                                    + phi[ip1_jm1] + phi[im1_jm1];

                    // 9‑point update
                    const double newPhi = (4.0*sumOrtho + sumDiag + 6.0*rho_q[idx]) / 20.0;
                    const double err    = fabs(newPhi - phi[idx]);
                    phi[idx]      = newPhi;

                    if (err > maxErrSweep) maxErrSweep = err;
                }
            }

            // combine sweep error into global error
            if (maxErrSweep > maxErr) maxErr = maxErrSweep;
        }

        // check convergence
        if (maxErr < tol) {
            break;
        }
    }
}
//──────────────────────────────────────────────────────────────────────────────
//  Poisson solver: 9-point stencil when BC are periodic for consistency.
//──────────────────────────────────────────────────────────────────────────────
void SolvePoisson_9point_Periodic(const std::vector<double>& rho_q,
                                  const int NX, const int NY){

    // Main GS iteration with 4‑color ordering
    for (size_t iter = 0; iter < maxIter; ++iter) {
        double maxErr = 0.0;

        // Perform one sweep for each of the 4 colors
        for (std::uint8_t sweep = 0; sweep < 4; ++sweep) {
            double maxErrSweep = 0.0;

            #pragma omp parallel for collapse(2) reduction(max:maxErrSweep) schedule(static)
            for (int j = 0; j < NY; ++j) {
                for (int i = 0; i < NX; ++i) {
                    // determine 4‑color index: 2*(i%2) + (j%2) in {0,1,2,3}
                    if ((2 * (i & 1) + (j & 1)) != sweep) 
                        continue;

                    // periodic neighbor indices
                    const int ip = (i + 1) % NX;
                    const int im = (i + NX - 1) % NX;
                    const int jp = (j + 1) % NY;
                    const int jm = (j + NY - 1) % NY;

                    // flattened indices
                    const int idx   = INDEX(i,  j,NX);
                    const int idxE  = INDEX(ip, j,NX);
                    const int idxW  = INDEX(im, j,NX);
                    const int idxN  = INDEX(i,  jp,NX);
                    const int idxS  = INDEX(i,  jm,NX);
                    const int idxNE = INDEX(ip, jp,NX);
                    const int idxNW = INDEX(im, jp,NX);
                    const int idxSE = INDEX(ip, jm,NX);
                    const int idxSW = INDEX(im, jm,NX);

                    // orthogonal and diagonal neighbor sums
                    const double sumO = phi[idxE]  + phi[idxW]
                                + phi[idxN]  + phi[idxS];
                    const double sumD = phi[idxNE] + phi[idxNW]
                                + phi[idxSE] + phi[idxSW];

                    // 9‑point Gauss–Seidel update
                    const double newPhi = (4.0 * sumO + sumD + 6.0 * rho_q[idx]) * 0.05; // /20
                    const double err    = fabs(newPhi - phi[idx]);
                    phi[idx]      = newPhi;

                    if (err > maxErrSweep) maxErrSweep = err;
                }
            }

            // accumulate the worst error across all sweeps
            if (maxErrSweep > maxErr) maxErr = maxErrSweep;
        }

        // convergence check
        if (maxErr < tol) {
            break;
        }
    }
}

//──────────────────────────────────────────────────────────────────────────────
// reconstruct E = –∇φ via central differences
//──────────────────────────────────────────────────────────────────────────────
void ComputeElectricField(std::vector<double>& Ex,
                          std::vector<double>& Ey,
                          const int NX, const int NY){
    // Compute electric field E = -∇φ using central differences
    // Only interior points; boundaries will be set by Neumann BC below
    #pragma omp parallel for collapse(2)
    for (int j = 1; j < NY - 1; ++j) {
        for (int i = 1; i < NX - 1; ++i) {
            const int idx = INDEX(i, j,NX);
            Ex[idx] = -0.5 * (phi[INDEX(i+1, j,NX)] - phi[INDEX(i-1, j,NX)]);
            Ey[idx] = -0.5 * (phi[INDEX(i, j+1,NX)] - phi[INDEX(i, j-1,NX)]);
        }
    }

    // Zero-Neumann boundary conditions (zero normal derivative):
    // copy the adjacent interior field value to the boundary cell.

    // Top and bottom boundaries
    #pragma omp parallel for
    for (int i = 0; i < NX; ++i) {
        Ex[INDEX(i, 0,NX)]      = Ex[INDEX(i, 1,NX)];
        Ey[INDEX(i, 0,NX)]      = Ey[INDEX(i, 1,NX)];
        Ex[INDEX(i, NY-1,NX)]   = Ex[INDEX(i, NY-2,NX)];
        Ey[INDEX(i, NY-1,NX)]   = Ey[INDEX(i, NY-2,NX)];
    }

    // Left and right boundaries
    #pragma omp parallel for
    for (int j = 0; j < NY; ++j) {
        Ex[INDEX(0, j,NX)]      = Ex[INDEX(1, j,NX)];
        Ey[INDEX(0, j,NX)]      = Ey[INDEX(1, j,NX)];
        Ex[INDEX(NX-1, j,NX)]   = Ex[INDEX(NX-2, j,NX)];
        Ey[INDEX(NX-1, j,NX)]   = Ey[INDEX(NX-2, j,NX)];
    }
}
//──────────────────────────────────────────────────────────────────────────────
// reconstruct E = –∇φ via central differences using Periodic BC
//──────────────────────────────────────────────────────────────────────────────
void ComputeElectricField_Periodic(std::vector<double>& Ex,
                                   std::vector<double>& Ey,
                                   const int NX, const int NY){
    // Compute electric field from potential using central differences
    // Periodic boundaries are handled with modulo indexing
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < NY; ++j) {
        for (int i = 0; i < NX; ++i) {
            const int im1 = (i + NX - 1) % NX;
            const int ip1 = (i + 1) % NX;
            const int jm1 = (j + NY - 1) % NY;
            const int jp1 = (j + 1) % NY;
            const int idx = INDEX(i, j,NX);

            Ex[idx] = -0.5 * (phi[INDEX(ip1, j,NX)] - phi[INDEX(im1, j,NX)]);
            Ey[idx] = -0.5 * (phi[INDEX(i, jp1,NX)] - phi[INDEX(i, jm1,NX)]);
        }
    }
}
//──────────────────────────────────────────────────────────────────────────────
// Define FFT objects
//──────────────────────────────────────────────────────────────────────────────
void InitPoissonFFT(const int NX, const int NY, const int NY_half, const int real_size) {
    const int complex_size = NX * NY_half; //size of complex space

    // Allocate FFTW arrays
    // In order to solve a probelam in the Fourier space we need to convert things in the frequency domani, then solve and the convert back 
    fft_in = (double*) fftw_malloc(sizeof(double) * real_size); //rho array in the real space
    fft_rho_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * complex_size); //rho array in the complex space
    fft_phi_hat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * complex_size); //phi array in the complex space
    fft_out = (double*) fftw_malloc(sizeof(double) * real_size); //phi array in the real space

    fft_plan_r2c = fftw_plan_dft_r2c_2d(NX, NY, fft_in, fft_rho_hat, FFTW_ESTIMATE); //plane for the transformation from rho real to rho complex
    fft_plan_c2r = fftw_plan_dft_c2r_2d(NX, NY, fft_phi_hat, fft_out, FFTW_ESTIMATE);//plane for the transformation from phi complex to phi real 
}
//──────────────────────────────────────────────────────────────────────────────
// Destroy for FFT objects
//──────────────────────────────────────────────────────────────────────────────
void FinalizePoissonFFT() {
    // Clean up FFTW allocations and plans
    fftw_destroy_plan(fft_plan_r2c);
    fftw_destroy_plan(fft_plan_c2r);
    fftw_free(fft_in);
    fftw_free(fft_out);
    fftw_free(fft_rho_hat);
    fftw_free(fft_phi_hat);
}

} // namespace poisson
