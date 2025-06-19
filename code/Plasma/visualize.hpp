// visualize.hpp
#pragma once

#include "utils.hpp"

#include <vector>
#include <utility>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>

namespace visualize {

//---------------------------------------------------------------------------
// Constants and sample points
//---------------------------------------------------------------------------

// Number of sample points (center + 8 around)
static constexpr int P = 9;

// Holds the (i,j) indices of the P sampling positions
extern std::array<std::pair<int,int>, P> sample_points;

//---------------------------------------------------------------------------
// Time-series buffers
//---------------------------------------------------------------------------

// Buffers: [time][point]
extern std::vector<std::array<double,P>> ts_ux_e, ts_uy_e, ts_ue_mag;
extern std::vector<std::array<double,P>> ts_ux_i, ts_uy_i, ts_ui_mag;
extern std::vector<std::array<double,P>> ts_ux_n, ts_uy_n, ts_un_mag;
extern std::vector<std::array<double,P>> ts_T_e,  ts_T_i,  ts_T_n;
extern std::vector<std::array<double,P>> ts_rho_e,ts_rho_i,ts_rho_n,ts_rho_q;
extern std::vector<std::array<double,P>> ts_Ex,   ts_Ey,    ts_E_mag;


//---------------------------------------------------------------------------
// OpenCV Video writers and configurations
//---------------------------------------------------------------------------
extern cv::VideoWriter video_writer_density;
extern cv::VideoWriter video_writer_velocity;
extern cv::VideoWriter video_writer_temperature;


//---------------------------------------------------------------------------
// Initialization Update and Finalization
//---------------------------------------------------------------------------

// Prepare sample_points and allocate all ts_* buffers for T steps
void InitVisualization(const int NX,const int NY, const int T);

// Record one time-step worth of data into in-memory buffers + render frames
void UpdateVisualization(const int t, const int NX, const int NY,
    const std::vector<double>& ux_e,  const std::vector<double>& uy_e,
    const std::vector<double>& ux_i,  const std::vector<double>& uy_i,
    const std::vector<double>& ux_n,  const std::vector<double>& uy_n,
    const std::vector<double>& T_e,   const std::vector<double>& T_i,
    const std::vector<double>& T_n,
    const std::vector<double>& rho_e, const std::vector<double>& rho_i,
    const std::vector<double>& rho_n, const std::vector<double>& rho_q,
    const std::vector<double>& Ex,    const std::vector<double>& Ey);


// After the run, plot all time-series buffers to PNG and release videos
void CloseVisualization();

//---------------------------------------------------------------------------
// Internal plotting helper
//---------------------------------------------------------------------------

// Plot one 2D time-series buffer into a PNG image using OpenCV
void PlotTimeSeriesWithOpenCV(const std::vector<std::array<double,P>>& data,
                              const std::string& title,
                              const std::string& png_filename);
//---------------------------------------------------------------------------
// Visualization routines
//---------------------------------------------------------------------------

// Render density and charge maps into the density video stream
void VisualizationDensity(const int NX, const int NY,
    const std::vector<double>& rho_e, const std::vector<double>& rho_i,
    const std::vector<double>& rho_q);

// Render velocity components and magnitude into the velocity video stream
void VisualizationVelocity(const int NX, const int NY,
    const std::vector<double>& ux_e,  const std::vector<double>& uy_e,
    const std::vector<double>& ux_i,  const std::vector<double>& uy_i);

// Render electron and ion temperature maps into the temperature video stream
void VisualizationTemperature(const int NX, const int NY,
    const std::vector<double>& T_e,   const std::vector<double>& T_i, const std::vector<double>& T_n);

//---------------------------------------------------------------------------
// Helper for the Videos
//---------------------------------------------------------------------------
cv::Mat normalize_and_color(const cv::Mat& src, const double vmin, const double vmax);
cv::Mat wrap_with_label(const cv::Mat& img, const std::string& label);

} // namespace visualize
