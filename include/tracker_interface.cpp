#include"./tracker_interface.h"
#include"./tracking_algo/staple_tracker.hpp"


ObjTracker::ObjTracker(std::map<std::string, double> params){
    // this->fps = fps;
    // this->tracker是void指针， void指针可以指向任何类型的变量
    this->tracker = new STAPLE_TRACKER();
    
    STAPLE_TRACKER *pTracker = (STAPLE_TRACKER *) this->tracker;
    for(std::map<std::string, double>::iterator it = params.begin(); it != params.end(); it++){
        const char* key = it->first.c_str();
        double val = it->second;

        if(strcmp(key, "use_color_feature") == 0)
            pTracker->cfg.use_color_feature = (val == 1.0 ? true : false);
        else if(strcmp(key, "use_color_hist_model") == 0)
            pTracker->cfg.use_color_hist_model = (val == 1.0 ? true : false);
        else if(strcmp(key, "use_apce_jugment") == 0)
            pTracker->cfg.use_apce_jugment = (val == 1.0 ? true : false);
        else if(strcmp(key, "use_part_track") == 0)
            pTracker->cfg.use_part_track = (val == 1.0 ? true : false);
        else if(strcmp(key, "fixed_area") == 0)
            pTracker->cfg.fixed_area = val;
        else if(strcmp(key, "fixed_area_part") == 0)
            pTracker->cfg.fixed_area_part = val;
        else if(strcmp(key, "scale_model_max_area") == 0)
            pTracker->cfg.scale_model_max_area = val;
        else if(strcmp(key, "num_scales") == 0)
            pTracker->cfg.num_scales = val;
        else if(strcmp(key, "scale_step") == 0)
            pTracker->cfg.scale_step = val;
        else if(strcmp(key, "merge_factor_cf") == 0)
            pTracker->cfg.merge_factor_cf = val;
        else if(strcmp(key, "merge_factor_scale") == 0)
            pTracker->cfg.merge_factor_scale = val;
        else if(strcmp(key, "learning_rate_cf") == 0)
            pTracker->cfg.learning_rate_cf = val;
        else if(strcmp(key, "learning_rate_cf_part") == 0)
            pTracker->cfg.learning_rate_cf_part = val;
        else if(strcmp(key, "learning_rate_pwp") == 0)
            pTracker->cfg.learning_rate_pwp = val;
        else if(strcmp(key, "learning_rate_scale") == 0)
            pTracker->cfg.learning_rate_scale = val;
        else if(strcmp(key, "part_detect_valid_thresh") == 0)
            pTracker->cfg.part_detect_valid_thresh = val;
        else if(strcmp(key, "beta1") == 0)
            pTracker->cfg.beta1 = val;
        else if(strcmp(key, "beta2") == 0)
            pTracker->cfg.beta2 = val;
    }
}


ObjTracker::~ObjTracker(){
    // this->work_ = false;
    // this->child_thread_.join();
    STAPLE_TRACKER *pTracker = (STAPLE_TRACKER *) this->tracker;
    delete pTracker;
    pTracker = nullptr;
    this->tracker = nullptr;
}


void ObjTracker::InitObjTracker(Mat &init_frame,const Rect &bbox){
    // this->work_ = false;
    // this->cur_box_ = bbox;
    Mat tracker_input_frame = init_frame;
    if(init_frame.channels() == 1)
        cv::cvtColor(init_frame, tracker_input_frame, cv::COLOR_GRAY2BGR);
    // 在使用时，需要先将void指针强制转化成对应类型的指针再使用
    STAPLE_TRACKER *pTracker = (STAPLE_TRACKER *) this->tracker;
    pTracker->tracker_staple_initialize(tracker_input_frame, bbox);
    pTracker->tracker_staple_train(tracker_input_frame, true);
}


Rect ObjTracker::TrackNextFrame(const Mat &new_frame, cv::Rect_<float> &part_rect1, cv::Rect_<float> &part_rect2){
    Mat tracker_input_frame = new_frame;
    if(new_frame.channels() == 1)
        cv::cvtColor(new_frame, tracker_input_frame, cv::COLOR_GRAY2BGR);
    // 在使用时，需要先将void指针强制转化成对应类型的指针再使用
    STAPLE_TRACKER *pTracker = (STAPLE_TRACKER *) this->tracker;

    double start = cv::getTickCount();
    Rect new_bbox = pTracker->tracker_staple_update(tracker_input_frame, part_rect1, part_rect2);
    pTracker->tracker_staple_train(tracker_input_frame, false);
    double end = cv::getTickCount();    
    std::cout << "fps:" << cv::getTickFrequency() / (end - start) << std::endl;

    return new_bbox;
}


// Rect ObjTracker::TrackNextFrame(const Mat &new_frame){
//     this->cur_frame_ = new_frame;
//     if(!this->work_){
//         this->work_ = true;
//         this->child_thread_ = std::thread(&ObjTracker::ChildThreadWork, this);
//     }
//     cv::waitKey(1000 / this->fps);
//     return this->cur_box_;
// }


// void ObjTracker::ChildThreadWork(){
//     STAPLE_TRACKER *pTracker = (STAPLE_TRACKER *) this->tracker;
//     while(this->work_){
//         cv::Mat input_frame = this->cur_frame_;
//         double start = cv::getTickCount();
//         this->cur_box_ = pTracker->tracker_staple_update(input_frame);
//         pTracker->tracker_staple_train(input_frame, false);
//         double end = cv::getTickCount();    
//         // std::cout << "fps: " << cv::getTickFrequency() / (end - start) << std::endl;
//     }
//     // this->work_ = false;
// }