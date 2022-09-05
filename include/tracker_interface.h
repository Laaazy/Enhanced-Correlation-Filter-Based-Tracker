#ifndef TRACKER_INTERFACE_H
#define TRACKER_INTERFACE_H

#include<thread>
#include<map>
#include<unordered_map>
#include<opencv2/opencv.hpp>

using cv::Mat;
using cv::Rect;

class ObjTracker{
public:
    void *tracker;

    ObjTracker(std::map<std::string, double> params);
    ~ObjTracker();

    void InitObjTracker(Mat &init_frame,const Rect &bbox);
    Rect TrackNextFrame(const Mat &new_frame, cv::Rect_<float> &part_rect1, cv::Rect_<float> &part_rect2);

private:
    // std::mutex mutex_;
    // std::thread child_thread_;
    // volatile bool work_;

    // Mat cur_frame_;
    // Rect cur_box_;
    
    // int fps;

    // void ChildThreadWork();
};

#endif