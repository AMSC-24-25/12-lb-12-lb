// visualize.cpp
#include "visualize.hpp"

#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace visualize {

//---------------------------------------------------------------------------
// Global data definitions
//---------------------------------------------------------------------------

// sample_points computed once in InitVisualization
std::array<std::pair<int,int>, P> sample_points;

// time-series buffers (allocated in InitVisualization)
std::vector<std::array<double,P>> ts_ux_e, ts_uy_e, ts_ue_mag,
                                  ts_ux_i, ts_uy_i, ts_ui_mag,
                                  ts_ux_n, ts_uy_n, ts_un_mag,
                                  ts_T_e,  ts_T_i,  ts_T_n,
                                  ts_rho_e,ts_rho_i,ts_rho_n,ts_rho_q,
                                  ts_Ex,   ts_Ey,    ts_E_mag;

cv::VideoWriter video_writer_density,
                video_writer_velocity,
                video_writer_temperature;

//---------------------------------------------------------------------------
// Visualization ranges (internal constants)
//---------------------------------------------------------------------------

constexpr double DENSITY_E_MIN = 0.0,    DENSITY_E_MAX = 1.0;
constexpr double DENSITY_I_MIN = 0.0,    DENSITY_I_MAX = 1822.0;
constexpr double CHARGE_MIN    = 0.0,    CHARGE_MAX    = 1.5;

constexpr double UX_E_MIN = -1e-7, UX_E_MAX = 1e-7;
constexpr double UY_E_MIN = -1e-7, UY_E_MAX = 1e-7;
constexpr double UE_MAG_MIN = 0.0, UE_MAG_MAX = 1e-7;

constexpr double UX_I_MIN = -1e-7, UX_I_MAX = 1e-7;
constexpr double UY_I_MIN = -1e-7, UY_I_MAX = 1e-7;
constexpr double UI_MAG_MIN = 0.0, UI_MAG_MAX = 1e-7;

constexpr double TEMP_E_MIN = 0.0, TEMP_E_MAX = 1.0;
constexpr double TEMP_I_MIN = 0.0, TEMP_I_MAX = 0.5;
constexpr double TEMP_N_MIN = 0.0, TEMP_N_MAX = 0.5;

// --- Global parameters for video ---
constexpr int border        = 10;
constexpr int label_height  = 30;

constexpr int tile_w        = 2 * border; //+NX
constexpr int tile_h        = 2 * border + label_height;//+NY

constexpr double fps        = 1.0;

// --- frame dimensions --- //We need to change them if the number of grid changes
constexpr int frame_w_density     = 3 * tile_w;//+3*NX
constexpr int frame_h_density     = tile_h;//+NY

constexpr int frame_w_velocity    = 3 * tile_w;//+3*NX
constexpr int frame_h_velocity    = 2 * tile_h;//+2*NY

constexpr int frame_w_temperature = 3 * tile_w; //+3*NX
constexpr int frame_h_temperature = tile_h;//+NY

//---------------------------------------------------------------------------
// InitVisualization
//---------------------------------------------------------------------------

void InitVisualization(const int NX, const int NY, const int T) {
    std::filesystem::create_directories("build/video");
    std::filesystem::create_directories("build/graphs");

    // Define sample points: center + 8 around
    const int cx = NX/2, cy = NY/2, dx = NX/4, dy = NY/4;
    sample_points = {{
        {cx, cy},
        {cx+dx, cy}, {cx-dx, cy},
        {cx, cy+dy}, {cx, cy-dy},
        {cx+dx, cy+dy}, {cx+dx, cy-dy},
        {cx-dx, cy+dy}, {cx-dx, cy-dy}
    }};

    // Allocate all time-series buffers
    ts_ux_e .resize(T, {});
    ts_uy_e .resize(T, {});
    ts_ue_mag.resize(T, {});
    ts_ux_i .resize(T, {});
    ts_uy_i .resize(T, {});
    ts_ui_mag.resize(T, {});
    ts_ux_n .resize(T, {});
    ts_uy_n .resize(T, {});
    ts_un_mag.resize(T, {});
    ts_T_e  .resize(T, {});
    ts_T_i  .resize(T, {});
    ts_T_n  .resize(T, {});
    ts_rho_e.resize(T, {});
    ts_rho_i.resize(T, {});
    ts_rho_n.resize(T, {});
    ts_rho_q.resize(T, {});
    ts_Ex   .resize(T, {});
    ts_Ey   .resize(T, {});
    ts_E_mag.resize(T, {});


    // Density
    video_writer_density.open(
        "build/video/video_density.mp4",
        cv::VideoWriter::fourcc('m','p','4','v'),
        fps,
        cv::Size(frame_w_density+3*NX, frame_h_density+NY),
        true
    );
    if (!video_writer_density.isOpened()) {
        std::cerr << "Cannot open video_density.mp4 for writing\n";
        return;
    }

    // Velocity
    video_writer_velocity.open(
        "build/video/video_velocity.mp4",
        cv::VideoWriter::fourcc('m','p','4','v'),
        fps,
        cv::Size(frame_w_velocity+3*NX, frame_h_velocity+2*NY),
        true
    );
    if (!video_writer_velocity.isOpened()) {
        std::cerr << "Cannot open video_velocity.mp4 for writing\n";
        return;
    }

    // Temperature
    video_writer_temperature.open(
        "build/video/video_temperature.mp4",
        cv::VideoWriter::fourcc('m','p','4','v'),
        fps,
        cv::Size(frame_w_temperature+3*NX, frame_h_temperature+NY),
        true
    );
    if (!video_writer_temperature.isOpened()) {
        std::cerr << "Cannot open video_temperature.mp4 for writing\n";
        return;
    }
}

//---------------------------------------------------------------------------
// Per-step update + recording
//---------------------------------------------------------------------------

void UpdateVisualization(const int t, const int NX, const int NY,
    const std::vector<double>& ux_e,  const std::vector<double>& uy_e,
    const std::vector<double>& ux_i,  const std::vector<double>& uy_i,
    const std::vector<double>& ux_n,  const std::vector<double>& uy_n,
    const std::vector<double>& T_e,   const std::vector<double>& T_i,
    const std::vector<double>& T_n,
    const std::vector<double>& rho_e, const std::vector<double>& rho_i,
    const std::vector<double>& rho_n, const std::vector<double>& rho_q,
    const std::vector<double>& Ex,    const std::vector<double>& Ey
) {
    // 1) Render videos
    VisualizationDensity(NX, NY, rho_e, rho_i, rho_q);
    VisualizationVelocity(NX, NY, ux_e, uy_e, ux_i, uy_i);
    VisualizationTemperature(NX, NY, T_e, T_i, T_n);

    // 2) Record into buffers
    auto& ux_ev  = ts_ux_e [t];
    auto& uy_ev  = ts_uy_e [t];
    auto& ue_mag = ts_ue_mag[t];
    auto& ux_iv  = ts_ux_i [t];
    auto& uy_iv  = ts_uy_i [t];
    auto& ui_mag = ts_ui_mag[t];
    auto& ux_nv  = ts_ux_n [t];
    auto& uy_nv  = ts_uy_n [t];
    auto& un_mag = ts_un_mag[t];
    auto& Te_v   = ts_T_e  [t];
    auto& Ti_v   = ts_T_i  [t];
    auto& Tn_v   = ts_T_n  [t];
    auto& re_v   = ts_rho_e[t];
    auto& ri_v   = ts_rho_i[t];
    auto& rn_v   = ts_rho_n[t];
    auto& rq_v   = ts_rho_q[t];
    auto& Ex_vv  = ts_Ex   [t];
    auto& Ey_vv  = ts_Ey   [t];
    auto& E_mag  = ts_E_mag[t];

    #pragma omp parallel for
    for (int p = 0; p < P; ++p) {
        const int i = sample_points[p].first;
        const int j = sample_points[p].second;
        const int idx = INDEX(i,j,NX);

        ux_ev[p] = ux_e[idx];
        uy_ev[p] = uy_e[idx];
        ue_mag[p]= std::sqrt(ux_ev[p]*ux_ev[p] + uy_ev[p]*uy_ev[p]);

        ux_iv[p] = ux_i[idx];
        uy_iv[p] = uy_i[idx];
        ui_mag[p]= std::sqrt(ux_iv[p]*ux_iv[p] + uy_iv[p]*uy_iv[p]);

        ux_nv[p] = ux_n[idx];
        uy_nv[p] = uy_n[idx];
        un_mag[p]= std::sqrt(ux_nv[p]*ux_nv[p] + uy_nv[p]*uy_nv[p]);

        Te_v[p]  = T_e[idx];
        Ti_v[p]  = T_i[idx];
        Tn_v[p]  = T_n[idx];

        re_v[p]  = rho_e[idx];
        ri_v[p]  = rho_i[idx];
        rn_v[p]  = rho_n[idx];
        rq_v[p]  = rho_q[idx];

        Ex_vv[p] = Ex[idx];
        Ey_vv[p] = Ey[idx];
        E_mag[p] = std::sqrt(Ex_vv[p]*Ex_vv[p] + Ey_vv[p]*Ey_vv[p]);
    }
}

//---------------------------------------------------------------------------
// Visualization rendering functions
//---------------------------------------------------------------------------

void VisualizationDensity(const int NX, const int NY,
                          const std::vector<double>& rho_e,
                          const std::vector<double>& rho_i,
                          const std::vector<double>& rho_q) {
    static cv::Mat mat_n_e(NY, NX, CV_32F), mat_n_i(NY, NX, CV_32F), mat_rho_q(NY, NX, CV_32F);

    #pragma omp parallel for collapse(2)
    for (int x = 0; x < NX; ++x)
        for (int y = 0; y < NY; ++y) {
            const int idx = INDEX(x, y, NX);
            mat_n_e.at<float>(y, x)   = static_cast<float>(rho_e[idx]);
            mat_n_i.at<float>(y, x)   = static_cast<float>(rho_i[idx]);
            mat_rho_q.at<float>(y, x) = static_cast<float>(rho_q[idx]);
        }

    cv::Mat grid;
    cv::hconcat(std::vector<cv::Mat>{
        wrap_with_label(normalize_and_color(mat_n_e, DENSITY_E_MIN, DENSITY_E_MAX), "rho_e"),
        wrap_with_label(normalize_and_color(mat_rho_q, CHARGE_MIN, CHARGE_MAX), "rho_q"),
        wrap_with_label(normalize_and_color(mat_n_i, DENSITY_I_MIN, DENSITY_I_MAX), "rho_i")
    }, grid);

    video_writer_density.write(grid);
}

void VisualizationVelocity(const int NX, const int NY,
                           const std::vector<double>& ux_e,
                           const std::vector<double>& uy_e,
                           const std::vector<double>& ux_i,
                           const std::vector<double>& uy_i) {
    static cv::Mat ux_e_mat(NY, NX, CV_32F), uy_e_mat(NY, NX, CV_32F), ue_mag(NY, NX, CV_32F);
    static cv::Mat ux_i_mat(NY, NX, CV_32F), uy_i_mat(NY, NX, CV_32F), ui_mag(NY, NX, CV_32F);

    #pragma omp parallel for collapse(2)
    for (int x = 0; x < NX; ++x)
        for (int y = 0; y < NY; ++y) {
            const int idx = INDEX(x, y, NX);
            double ux_el = ux_e[idx], uy_el = uy_e[idx];
            double ux_ion = ux_i[idx], uy_ion = uy_i[idx];
            ux_e_mat.at<float>(y, x) = static_cast<float>(ux_el);
            uy_e_mat.at<float>(y, x) = static_cast<float>(uy_el);
            ue_mag.at<float>(y, x)   = static_cast<float>(std::sqrt(ux_el * ux_el + uy_el * uy_el));
            ux_i_mat.at<float>(y, x) = static_cast<float>(ux_ion);
            uy_i_mat.at<float>(y, x) = static_cast<float>(uy_ion);
            ui_mag.at<float>(y, x)   = static_cast<float>(std::sqrt(ux_ion * ux_ion + uy_ion * uy_ion));
        }

    cv::Mat top, bot, grid;
    cv::hconcat(std::vector<cv::Mat>{
        wrap_with_label(normalize_and_color(ux_e_mat, UX_E_MIN, UX_E_MAX), "ux_e"),
        wrap_with_label(normalize_and_color(uy_e_mat, UY_E_MIN, UY_E_MAX), "uy_e"),
        wrap_with_label(normalize_and_color(ue_mag, UE_MAG_MIN, UE_MAG_MAX), "|u_e|")
    }, top);

    cv::hconcat(std::vector<cv::Mat>{
        wrap_with_label(normalize_and_color(ux_i_mat, UX_I_MIN, UX_I_MAX), "ux_i"),
        wrap_with_label(normalize_and_color(uy_i_mat, UY_I_MIN, UY_I_MAX), "uy_i"),
        wrap_with_label(normalize_and_color(ui_mag, UI_MAG_MIN, UI_MAG_MAX), "|u_i|")
    }, bot);

    cv::vconcat(top, bot, grid);

    video_writer_velocity.write(grid);
}

void VisualizationTemperature(const int NX, const int NY,
                              const std::vector<double>& T_e,
                              const std::vector<double>& T_i,
                              const std::vector<double>& T_n) {
    static cv::Mat Te_mat(NY, NX, CV_32F), Ti_mat(NY, NX, CV_32F), Tn_mat(NY, NX, CV_32F);

    #pragma omp parallel for collapse(2)
    for (int x = 0; x < NX; ++x)
        for (int y = 0; y < NY; ++y) {
            const int idx = INDEX(x, y, NX);
            Te_mat.at<float>(y, x) = static_cast<float>(T_e[idx]);
            Ti_mat.at<float>(y, x) = static_cast<float>(T_i[idx]);
            Tn_mat.at<float>(y, x) = static_cast<float>(T_n[idx]); // nuovo campo
        }

    cv::Mat grid;
    cv::hconcat(std::vector<cv::Mat>{
        wrap_with_label(normalize_and_color(Te_mat, TEMP_E_MIN, TEMP_E_MAX), "T_e"),
        wrap_with_label(normalize_and_color(Ti_mat, TEMP_I_MIN, TEMP_I_MAX), "T_i"),
        wrap_with_label(normalize_and_color(Tn_mat, TEMP_N_MIN, TEMP_N_MAX), "T_n")
    }, grid);

    video_writer_temperature.write(grid);
}
//---------------------------------------------------------------------------
// Helper for the Videos
//---------------------------------------------------------------------------
cv::Mat normalize_and_color(const cv::Mat& src, const double vmin, const double vmax) {
    cv::Mat norm, color; //norm is the normalized image in 8 bit //color is the image colored
    src.convertTo(norm, CV_8U, 255.0 / (vmax - vmin), -vmin * 255.0 / (vmax - vmin)); //this normalize once maximum and miinimum value is given
    cv::applyColorMap(norm, color, cv::COLORMAP_JET); //this apply the Jet color
    cv::flip(color, color, 0); //this is because the y should be down
    return color;
}

cv::Mat wrap_with_label(const cv::Mat& img, const std::string& label) { //starting from img that is colored from before and the label chosen
    cv::Mat bordered; //this will be the image with borders
    cv::copyMakeBorder(img, bordered, border, border + label_height, border, border,
                       cv::BORDER_CONSTANT, cv::Scalar(255,255,255)); //this add borders
    cv::putText(bordered, label, cv::Point(border + 5, bordered.rows - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 1);  //this write the label
    return bordered;
}
//---------------------------------------------------------------------------
// Finalization
//---------------------------------------------------------------------------

void CloseVisualization() {
    // 1) Create plots
    // Create box to images
    std::vector<std::pair<cv::Mat, std::string>> plots;
    // Put all the images in the box
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_ux_e,  "ux_e"),   "plot_ux_e.png"});
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_uy_e,  "uy_e"),   "plot_uy_e.png"});
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_ue_mag,"|u_e|"), "plot_ue_mag.png"});
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_ux_i,  "ux_i"),   "plot_ux_i.png"});
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_uy_i,  "uy_i"),   "plot_uy_i.png"});
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_ui_mag,"|u_i|"), "plot_ui_mag.png"});
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_ux_n,  "ux_n"),   "plot_ux_n.png"});
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_uy_n,  "uy_n"),   "plot_uy_n.png"});
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_un_mag,"|u_n|"), "plot_un_mag.png"});
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_T_e,   "T_e"),    "plot_T_e.png"});
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_T_i,   "T_i"),    "plot_T_i.png"});
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_T_n,   "T_n"),    "plot_T_n.png"});
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_rho_e, "rho_e"),  "plot_rho_e.png"});
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_rho_i, "rho_i"),  "plot_rho_i.png"});
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_rho_n, "rho_n"),  "plot_rho_n.png"});
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_rho_q, "rho_q"),  "plot_rho_q.png"});
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_Ex,    "Ex"),     "plot_Ex.png"});
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_Ey,    "Ey"),     "plot_Ey.png"});
    plots.push_back({PlotTimeSeriesWithOpenCV(ts_E_mag, "|E|"),    "plot_E_mag.png"});
    //Plot it one time to save time
    #pragma omp parallel for
    for (size_t i = 0; i < plots.size(); ++i) {
        const auto& [img, name] = plots[i];
        if (!img.empty())
            cv::imwrite("build/graphs/" + name, img);
    }

    // 2) Release video writers
    video_writer_density.release();
    video_writer_velocity.release();
    video_writer_temperature.release();
    std::cout << "Videos and Plots generated.\n";
}

//---------------------------------------------------------------------------
// PlotTimeSeriesWithOpenCV
//---------------------------------------------------------------------------

cv::Mat PlotTimeSeriesWithOpenCV(
    const std::vector<std::array<double,P>>& data,
    const std::string& title
) {
    if (data.empty()) return cv::Mat(); //if empty return
    const int N = data.size(); //take the size of the data inside

    // Find global min/max over all points to plot and see data
    double vmin = data[0][0], vmax = data[0][0];
    #pragma omp parallel for reduction(min:vmin) reduction(max:vmax)
    for (const auto& arr : data)
        for (double v : arr) {
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
    if (vmin == vmax) { vmin -= 1.0; vmax += 1.0; } // if constant value just add some space to avoid seeing line grafics

    // Create plotting canvas
    const int W = 800, H = 600; //image size
    const int ml = 80, mr = 40, mt = 60, mb = 80; //additiona space for axis and labels
    cv::Mat img(H, W, CV_8UC3, cv::Scalar::all(255)); //define empty image

    // Draw axes
    cv::Point origin(ml, H-mb), xend(W-mr, H-mb), yend(ml, mt);
    cv::line(img, origin, xend, cv::Scalar(0,0,0));
    cv::line(img, origin, yend, cv::Scalar(0,0,0));
    cv::putText(img, title, {ml, mt/2},
                cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,0,0}, 2);

    const double pw = W - ml - mr, ph = H - mt - mb; //total graph width and height
    const double xs = pw / (N - 1), ys = ph / (vmax - vmin); //x and y steps

    // Color palette
    std::array<cv::Scalar, P> palette;
    std::vector<cv::Scalar> base_palette = {
        {255,0,0},{0,128,0},{0,0,255},{255,165,0},
        {128,0,128},{0,255,255},{255,0,255},{128,128,0},
        {0,128,128}
    };
    for (int p = 0; p < P; ++p)
        palette[p] = base_palette[p % base_palette.size()];

    // Plot each of the P curves
    //#pragma omp parallel for
    for (int p = 0; p < P; ++p) {
        for (int k = 1; k < N; ++k) {
            //Point 0
            const int x0 = int(ml + xs * (k - 1));
            const int y0 = int(H - mb - ys * (data[k - 1][p] - vmin));
            //Point 1
            const int x1 = int(ml + xs * k);
            const int y1 = int(H - mb - ys * (data[k][p] - vmin));
            cv::line(img, {x0, y0}, {x1, y1}, palette[p], 1, cv::LINE_AA); //draw a line between point 0 and point 1
        }
    }
    // Legend
    const int lx = W - mr - 150, ly = mt + 10, lh = 20;
    for (int p = 0; p < P; ++p) {
        cv::rectangle(img,
            {lx, ly + p*lh}, {lx+15, ly + p*lh + 15},
            palette[p % palette.size()], cv::FILLED
        ); //draw a small Rect to see the color of the curve
        cv::putText(img,
            "p" + std::to_string(p),
            {lx+20, ly + p*lh + 12},
            cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,0,0}, 1
        ); //Tell wich point is
    }

    // Axis labels
    cv::putText(img, "time",
        {(ml + W-mr)/2, H-mb + 40},
        cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,0}, 1
    ); //x axis
    cv::putText(img, title,
        {10, (mt + H-mb)/2},
        cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,0}, 1
    ); // y axis

    return img; //return image
}




} // namespace visualize
