#include<opencv2/opencv.hpp>

using cv::Mat;

// calculate apce
void calAPCE(Mat response, double &curr_apce_val, double &curr_F_max){
    double min_val, max_val;
    cv::minMaxLoc(response, &min_val, &max_val, NULL, NULL);

    double min_max_diff = std::pow(max_val - min_val, 2);
    
    double sum = 0, ave = 0;
    for(int i = 0; i < response.rows; i++)
        for(int j = 0; j < response.cols; j++)
            sum += std::pow(response.at<float>(i, j) - min_val, 2);
    ave = sum / (response.rows * response.cols);

    curr_apce_val = min_max_diff / ave;
    curr_F_max = max_val;
}