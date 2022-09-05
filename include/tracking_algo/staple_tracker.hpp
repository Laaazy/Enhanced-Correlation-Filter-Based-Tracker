#ifndef STAPLE_TRACKER_HPP
#define STAPLE_TRACKER_HPP

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;

///
/// \brief The staple_cfg struct
///
struct staple_cfg{
    // --------------------------user config--------------------------------
    bool use_color_feature = true;
    bool use_color_hist_model = true;
    bool use_apce_jugment = true;
    bool use_part_track = true;

    int fixed_area = 150*150;               // standard area to which we resize the target
    int fixed_area_part = 75*75;            // max area of part track filters
    double scale_model_max_area = 32*16;    // max area of scale filter

    int num_scales = 33;                    // num of scales 
    double scale_step = 1.02;               // step size of scale

    double merge_factor_cf = 0.3;           // the interpolation factor of cf response and color model prob
    double merge_factor_scale = 0.25;       // the interpolation factor of cf and part based scale estimation
    
    double learning_rate_cf = 0.01;         // location filter learning rate
    double learning_rate_cf_part = 0.01;    // part location filter learning rate
    double learning_rate_pwp = 0.04;        // bg and fg color models learning rate
    double learning_rate_scale = 0.05;      // scale filter learning rate, Default staple=0.02
    
    double part_detect_valid_thresh = 0.554;
    double beta1 = 0.5;                     // threshold factor of apce value
    double beta2 = 0.5;                     // threshold factor of max response value
    // ----------------------------------------------------------------------

    bool grayscale_sequence = false;     // suppose that sequence is colour
    int hog_cell_size = 4;
    int n_bins = 2*2*2*2*2;              // number of bins for the color histograms (bg and fg models)
    const char * feature_type = "fhog";  // "fhog", ""gray""
    double inner_padding = 0.2;          // defines inner area used to sample colors from the foreground
    double output_sigma_factor = 1/16.0; // standard deviation for the desired translation filter output
    double lambda = 1e-3;                // egularization weight
    const char * merge_method = "const_factor";
    bool den_per_channel = false;

    // scale related
    bool scale_adaptation = true;
    int hog_scale_cell_size = 4;         // Default DSST=4
    double scale_sigma_factor = 1/4.0;
    double scale_model_factor = 1.0;
    
    // debugging stuff
    int visualization = 0;              // show output bbox on frame
    int visualization_dbg = 0;          // show also per-pixel scores, desired response and filter output

    cv::Point_<float> init_pos;
    cv::Size target_sz;
};

///
/// \brief The STAPLE_TRACKER class
///
class STAPLE_TRACKER{
public:
    staple_cfg cfg;
    STAPLE_TRACKER()
    {
        cfg = default_parameters_staple(cfg);
        frameno = 0;

        num_apce = 0;
        mean_apce = 0;
        mean_fmax = 0;
        apce_update = true;

        curr_scale_based_on_part = 1.0;
    }
    ~STAPLE_TRACKER(){}

    void mexResize(const cv::Mat &im, cv::Mat &output, cv::Size newsz, const char *method);
    void tracker_staple_train(cv::Mat im, bool first);
    void tracker_staple_initialize(cv::Mat im, cv::Rect_<float> region);
    cv::Rect tracker_staple_update(cv::Mat im, cv::Rect_<float> &part_rect1, cv::Rect_<float> &part_rect2);

protected:
    staple_cfg default_parameters_staple(staple_cfg cfg);
    void initializeAllAreas(const cv::Mat &im);

    void getSubwindow(const cv::Mat &im, cv::Point_<float> centerCoor, cv::Size model_sz, cv::Size scaled_sz, cv::Mat &output);
    void getSubwindowFloor(const cv::Mat &im, cv::Point_<float> centerCoor, cv::Size model_sz, cv::Size scaled_sz, cv::Mat &output);
    void updateHistModel(bool new_model, cv::Mat &patch, double learning_rate_pwp=0.0);
    void CalculateHann(cv::Size sz, cv::Mat &output);
    void gaussianResponse(cv::Size rect_size, double sigma, cv::Mat &output);
    void getFeatureMap(cv::Mat &im_patch, const char *feature_type, cv::MatND &output);
    void cropFilterResponse(const cv::Mat &response_cf, cv::Size response_size, cv::Mat& output);
    void getColourMap(const cv::Mat &patch, cv::Mat& output);
    void getCenterLikelihood(const cv::Mat &object_likelihood, cv::Size m, cv::Mat& center_likelihood);
    void mergeResponses(const cv::Mat &response_cf, const cv::Mat &response_pwp, cv::Mat &response);
    void getScaleSubwindow(const cv::Mat &im, cv::Point_<float> centerCoor, cv::Mat &output);

    void load_w2c();
    void getColorNameFeat(const cv::Mat &im, cv::Mat &output);

    void part_initialize(cv::Mat im, cv::Rect_<float> region);
    void part_train(cv::Mat im, bool is_first);
    void part_update(cv::Mat im, cv::Rect_<float> &part_rect1, cv::Rect_<float> &part_rect2);

private:
    cv::Point_<float> pos;
    cv::Size target_sz;

    cv::Size bg_area;
    cv::Size fg_area;
    double area_resize_factor;
    cv::Size cf_response_size;

    cv::Size norm_bg_area;
    cv::Size norm_target_sz;
    cv::Size norm_delta_area;
    cv::Size norm_pwp_search_area;

    cv::MatND bg_hist;
    cv::MatND fg_hist;

    cv::Mat hann_window;
    cv::Mat yf;

    std::vector<cv::Mat> hf_den;
    std::vector<cv::Mat> hf_num;

    // part track
    bool is_tall;
    std::vector<cv::Rect_<float>> rect_part;
    cv::Size bg_area_part;
    cv::Size norm_bg_area_part;
    cv::Size norm_target_sz_part;
    cv::Size norm_delta_area_part;
    double area_resize_factor_part;
    cv::Size cf_response_size_part;
    cv::Mat hann_window_part;
    cv::Mat yf_part;
    std::vector<std::vector<cv::Mat>> hf_num_part;
    std::vector<std::vector<cv::Mat>> hf_den_part;
    std::vector<bool> valid_part;
    double init_part_dist;
    double curr_scale_based_on_part;


    cv::Rect rect_position;

    float scale_factor;
    cv::Mat scale_window;
    cv::Mat scale_factors;
    cv::Size scale_model_sz;
    float min_scale_factor;
    float max_scale_factor;
    cv::Size base_target_sz;

    cv::Mat ysf;
    cv::Mat sf_den;
    cv::Mat sf_num;

    int frameno;

    // APCE
    long num_apce;
    double mean_apce;
    double mean_fmax;
    bool apce_update;

    vector<vector<double>> w2c;
};

#endif