#include "streaming.hpp"


#include <stdexcept>

namespace streaming {
// Opposite directions for bounce‐back:
constexpr std::array<int, Q> opp = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };

//──────────────────────────────────────────────────────────────────────────────
//  Streaming dispatcher:
//──────────────────────────────────────────────────────────────────────────────
void Stream(std::vector<double>& f_e, std::vector<double>& f_i, std::vector<double>& f_n, 
            std::vector<double>& temp_e, std::vector<double>& temp_i, std::vector<double>& temp_n, 
            std::vector<double>& g_e, std::vector<double>& g_i, std::vector<double>& g_n,
            const std::array<int, Q>& cx, const std::array<int, Q>& cy,
            const int NX, const int NY, const BCType type) {
    switch (type) {
        case BCType::Periodic:
            StreamingPeriodic(f_e, f_i, f_n, temp_e, temp_i, temp_n, cx, cy, NX, NY);
            ThermalStreamingPeriodic(g_e, g_i, g_n, temp_e, temp_i, temp_n, cx, cy, NX, NY);
            break;
        case BCType::BounceBack:
            StreamingBounceBack(f_e, f_i, f_n, temp_e, temp_i, temp_n, cx, cy, NX, NY);
            ThermalStreamingBounceBack(g_e, g_i, g_n, temp_e, temp_i, temp_n, cx, cy, NX, NY);
            break;
        default:
            throw std::runtime_error("Tipo di streaming non supportato.");
    }
}
//──────────────────────────────────────────────────────────────────────────────
//  Streaming + PERIODIC boundaries:
//    f_{i}(x + c_i, y + c_i, t+1) = f_{i}(x,y,t)
//──────────────────────────────────────────────────────────────────────────────
void StreamingPeriodic(std::vector<double>& f_e, std::vector<double>& f_i, std::vector<double>& f_n,
                       std::vector<double>& temp_e, std::vector<double>& temp_i, std::vector<double>& temp_n, 
                       const std::array<int, Q>& cx, const std::array<int, Q>& cy,
                       const int NX, const int NY) {
    #pragma omp parallel for collapse(3) schedule(static)
    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            for (int i = 0; i < Q; ++i) {
                // Streaming coordinates (periodic wrapping)
                const int x_str = (x + NX + cx[i]) % NX;
                const int y_str = (y + NY + cy[i]) % NY;

                const int idx_to    = INDEX(x_str, y_str, i,NX,Q);
                const int idx_from  = INDEX(x, y, i,NX,Q);

                temp_e[idx_to] = f_e[idx_from];
                temp_i[idx_to] = f_i[idx_from];
                temp_n[idx_to] = f_n[idx_from];
            }
        }
    }
    f_e.swap(temp_e);
    f_i.swap(temp_i);
    f_n.swap(temp_n);
}
//──────────────────────────────────────────────────────────────────────────────
//  Streaming + BOUNCE‐BACK walls at x=0, x=NX−1, y=0, y=NY−1.
//    If (x + c_i,x) or (y + c_i,y) is out‐of‐bounds, reflect:
//       f_{i*}(x,y) += f_{i}(x,y),
//    where i* = opp[i].
//──────────────────────────────────────────────────────────────────────────────
void StreamingBounceBack(std::vector<double>& f_e, std::vector<double>& f_i, std::vector<double>& f_n,
                       std::vector<double>& temp_e, std::vector<double>& temp_i, std::vector<double>& temp_n, 
                       const std::array<int, Q>& cx, const std::array<int, Q>& cy,
                       const int NX, const int NY) {
    #pragma omp for collapse(3) schedule(static)
        for (int x = 0; x < NX; ++x) {
            for (int y = 0; y < NY; ++y) {
                for (int i = 0; i < Q; ++i) {
                    //Define streaming coordinate
                    const int x_str = x + cx[i];
                    const int y_str = y + cy[i];
                    const int idx_from   = INDEX(x, y, i,NX,Q);
                    if (x_str >= 0 && x_str <NX && y_str >= 0 && y_str <NY) {
                        const int idx_to = INDEX(x_str, y_str, i,NX,Q);
                        //We are inside the lattice so simple streaming
                        temp_e[idx_to] = f_e[idx_from];
                        temp_i[idx_to] = f_i[idx_from];
                        temp_n[idx_to] = f_n[idx_from];
                    }
                    else if(x_str >= 0 && x_str <NX){
                        const int idx_to    = INDEX(x_str, y, opp[i],NX,Q);
                        //We are outside on y so bounceback only on y, x continues
                        temp_e[idx_to]=f_e[idx_from];
                        temp_i[idx_to]=f_i[idx_from];
                        temp_n[idx_to]=f_n[idx_from];
                    }
                    else if(y_str >= 0 && y_str <NY){
                        const int idx_to    = INDEX(x, y_str, opp[i],NX,Q);
                        //We are outside on x so bounceback only on x, y continues
                        temp_e[idx_to]=f_e[idx_from];
                        temp_i[idx_to]=f_i[idx_from];
                        temp_n[idx_to]=f_n[idx_from];
                    }
                    else{
                        const int idx_to    = INDEX(x, y, opp[i],NX,Q);
                        //We are outside on both sides so bounceback
                        temp_e[idx_to]=f_e[idx_from];
                        temp_i[idx_to]=f_i[idx_from];
                        temp_n[idx_to]=f_n[idx_from];
                    }
                }
            }
        }
    f_e.swap(temp_e);
    f_i.swap(temp_i);
    f_n.swap(temp_n);
}
//──────────────────────────────────────────────────────────────────────────────
//  Streaming + PERIODIC boundaries:
//    g_{i}(x + c_i, y + c_i, t+1) = g_{i}(x,y,t)l
//──────────────────────────────────────────────────────────────────────────────
void ThermalStreamingPeriodic(std::vector<double>& g_e, std::vector<double>& g_i, std::vector<double>& g_n,
                              std::vector<double>& temp_e, std::vector<double>& temp_i, std::vector<double>& temp_n, 
                              const std::array<int,Q>& cx, const std::array<int,Q>& cy,
                              const int NX, const int NY) {
    #pragma omp parallel for collapse(3) schedule(static)
    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            for (int i = 0; i < Q; ++i) {
                // Streaming coordinates (periodic wrapping)
                const int x_str = (x + NX + cx[i]) % NX;
                const int y_str = (y + NY + cy[i]) % NY;

                const int idx_to    = INDEX(x_str, y_str, i,NX,Q);
                const int idx_from  = INDEX(x, y, i,NX,Q);

                temp_e[idx_to]  = g_e[idx_from];
                temp_i[idx_to]  = g_i[idx_from];
                temp_n[idx_to]  = g_n[idx_from];
            }
        }
    }
    g_e.swap(temp_e);
    g_i.swap(temp_i);
    g_n.swap(temp_n);
}

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
                       const int NX, const int NY) {
    #pragma omp for collapse(3) schedule(static)
        for (int x = 0; x < NX; ++x) {
            for (int y = 0; y < NY; ++y) {
                for (int i = 0; i < Q; ++i) {
                    //Define streaming coordinate
                    const int x_str = x + cx[i];
                    const int y_str = y + cy[i];
                    const int idx_from   = INDEX(x, y, i,NX,Q);
                    if (x_str >= 0 && x_str <NX && y_str >= 0 && y_str <NY) {
                        const int idx_to = INDEX(x_str, y_str, i,NX,Q);
                        //We are inside the lattice so simple streaming
                        temp_e[idx_to] = g_e[idx_from];
                        temp_i[idx_to] = g_i[idx_from];
                        temp_n[idx_to] = g_n[idx_from];
                    }
                    else if(x_str >= 0 && x_str <NX){
                        const int idx_to = INDEX(x_str, y, opp[i],NX,Q);
                        //We are outside so bounceback
                        temp_e[idx_to]=g_e[idx_from];
                        temp_i[idx_to]=g_i[idx_from];
                        temp_n[idx_to]=g_n[idx_from];
                    }
                    else if(y_str >= 0 && y_str <NY){
                        const int idx_to = INDEX(x, y_str, opp[i],NX,Q);
                        //We are outside so bounceback
                        temp_e[idx_to]=g_e[idx_from];
                        temp_i[idx_to]=g_i[idx_from];
                        temp_n[idx_to]=g_n[idx_from];
                    }
                    else{
                        const int idx_to = INDEX(x, y, opp[i],NX,Q);
                        //We are outside so bounceback
                        temp_e[idx_to]=g_e[idx_from];
                        temp_i[idx_to]=g_i[idx_from];
                        temp_n[idx_to]=g_n[idx_from];
                    }
                }
            }
        }
    g_e.swap(temp_e);
    g_i.swap(temp_i);
    g_n.swap(temp_n);
}

}  // namespace streaming
