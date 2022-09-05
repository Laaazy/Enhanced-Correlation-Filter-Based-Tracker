#include<unistd.h>
#include<iostream>
#include<map>
#include<opencv2/opencv.hpp>
#include"include/tracker_interface.h"

using namespace cv;
using namespace std;

// int main(){
//     // --------------------------user config--------------------------------
//     double use_color_feature = true;        // 是否使用颜色特征
//     double use_color_hist_model = true;     // 是否使用颜色统计直方图
//     double use_apce_jugment = true;         // 是否使用apce判据，apce判据对于有遮挡等跟踪置信度低的场景有一定改善
//     double use_part_track = true;           // 是否开启基于部分目标和完整目标的local-global跟踪，开启时能改善位置和尺寸跟踪，也会降低跟踪速度

//     int fixed_area = 150*150;               // 目标区域的尺寸大小
//     int fixed_area_part = 75*75;            // 开启part_track时的部分目标区域尺寸大小
//     double scale_model_max_area = 32*16;    // 尺寸估计模型的目标区域大小

//     int num_scales = 33;                    // 尺寸估计的尺寸数量
//     double scale_step = 1.02;               // 尺寸估计的步长，默认情况每一帧的尺寸估计涵盖[1.02^-16, ..., 1.02^-1, 1, 1.02, ..., 1.02^16]这33种尺寸

//     double merge_factor_cf = 0.3;           // 使用颜色统计直方图时，基本特征和颜色直方图的合并系数，(1 - merge_factor_cf) * 基本特征 + merge_factor_cf * 颜色直方图
//     double merge_factor_scale = 0.25;       // 开启part_track时，基于part的尺寸估计结果和全局尺寸估计结果的合并系数，(1 - merge_factor_scale) * 全局尺寸估计结果 + merge_factor_scale * 基于part的尺寸估计结果
    
//     double learning_rate_cf = 0.01;         // 全局位置滤波器的学习率
//     double learning_rate_cf_part = 0.01;    // 开启part_track时，part位置滤波器的学习率
//     double learning_rate_pwp = 0.04;        // 使用颜色统计直方图时，颜色统计直方图的学习率
//     double learning_rate_scale = 0.05;      // 尺寸估计滤波器的学习率
    
//     double part_detect_valid_thresh = 0.554;// 开启part_track时，part滤波器的跟踪置信度阈值，若某帧跟踪置信度低于此阈值，则part滤波器的估计结果不作为全局跟踪的参考，且不会更新part滤波器，可选值范围[0.5, 1.0]
//     double beta1 = 0.5;                     // 使用apce判据时，当前帧apce值 > beta1 * 历史平均apce值才有机会更新整个tracker。适当的阈值可以避免有遮挡时，apce和置信度低的结果更新tracker。
//     double beta2 = 0.5;                     // 使用apce判据时，当前帧最大跟踪置信度 > beta2 * 历史平均最大跟踪置信度才有机会更新整个tracker。
//     // ----------------------------------------------------------------------

//     // 以上是可配置参数，需要时可在参数字典中添加对应的键值对进行调整
//     std::map<std::string, double> params{
//         {"use_color_feature", 0}, 
//         {"use_color_hist_model", 0},
//         {"use_apce_jugment", 0},
//         {"use_part_track", 0},
//         {"fixed_area", 75*75} // 例如这里调整目标区域的尺寸从150*150变为75*75，速度会很大提升，精度会有所下降
//     };
//     // 每次使用先实例化tracker对象
//     ObjTracker* tracker = new ObjTracker(params);

//     VideoCapture capture;
//     capture.open("./input/video/test_video2.mp4");
//     int numFrames = 0, cnt = 0;
//     if(capture.isOpened()){
//         numFrames = (int) capture.get(CAP_PROP_FRAME_COUNT);  
//         Mat frame;
//         int frameCount = 0;
//         while(true){
//             usleep(20000);
//             capture.read(frame);
//             cnt++;
//             imshow("video", frame);
//             char cmd = waitKey(10);
//             if(cmd == 's'){
//                 Rect bbox = selectROI(frame, false);
//                 // 初始化tracker
//                 tracker->InitObjTracker(frame, bbox);
//                 break;
//             }
//         }

//         while (cnt < numFrames){
//             capture.read(frame);
//             cnt++;
//             cv::Rect_<float> part_rect1, part_rect2;
//             // 跟踪新的帧,返回新帧中的物体位置
//             Rect bbox = tracker->TrackNextFrame(frame, part_rect1, part_rect2);

//             cv::Point_<float> part_center1(part_rect1.x + 0.5*part_rect1.width, part_rect1.y + 0.5*part_rect1.height);
//             cv::Point_<float> part_center2(part_rect2.x + 0.5*part_rect2.width, part_rect2.y + 0.5*part_rect2.height);
//             cv::Point_<float> rough_center(0.5*(part_center1.x + part_center2.x), 0.5*(part_center1.y + part_center2.y));
//             cv::Point_<float> final_center(bbox.x + 0.5*bbox.width, bbox.y + 0.5*bbox.height);

//             circle(frame, part_center1, 2, Scalar(255, 0, 0), -1);
//             circle(frame, part_center2, 2, Scalar(255, 0, 0), -1);
//             line(frame, part_center1, part_center2, Scalar(0, 0, 255), 1);
//             circle(frame, rough_center, 2, Scalar(255, 0, 0), -1);
//             // rectangle(frame, part_rect1, cv::Scalar(0, 255, 0), 1, 1);
//             // rectangle(frame, part_rect2, cv::Scalar(0, 255, 0), 1, 1);
//             circle(frame, final_center, 2, Scalar(0, 255, 0), -1);
//             rectangle(frame, bbox, cv::Scalar(0,255, 0), 1, 1);
            
//             imshow("video", frame);
//             if(waitKey(10) == 'q')
//                 break;
//             // std::cout << cnt << "/" << numFrames << std::endl;
//         }
//     }

//     // 不使用了需要delete释放内存空间
//     delete tracker;
//     return 0;
// }

int main(){
    std::map<std::string, std::vector<int>> map{
        {"Basketball", {198,214,34,81}},
        {"Biker", {262,94,16,26}},
        {"Bird1", {450,91,31,37}},
        {"BlurBody", {400,48,87,319}},
        {"BlurCar2", {227,207,122,99}},
        {"BlurFace", {246,226,94,114}},
        {"BlurOwl", {352,197,56,100}},
        {"Bolt", {336,165,26,61}},
        {"Box", {478,143,80,111}},
        {"Car1", {23,88,66,55}},
        {"Car4", {70,51,107,87}},
        {"CarDark", {73,126,29,23}},
        {"CarScale", {6,166,42,26}},
        {"ClifBar", {143,125,30,54}},
        {"Couple", {51,47,25,62}},
        {"Crowds", {561,311,22,51}},
        {"David", {130,60,115,90}},
        {"Deer", {306,5,95,65}},
        {"Diving", {177,51,21,129}},
        {"DragonBaby", {160,83,56,65}},
        {"Dudek", {123,87,132,176}},
        {"Football", {310,102,39,50}},
        {"Freeman4", {125,86,15,16}},
        {"Girl", {57,21,31,45}},
        {"Human3", {264,311,37,69}},
        {"Human4", {99,237,27,82}},
        {"Human6", {340,358,18,55}},
        {"Human9", {93,113,34,109}},
        {"Ironman", {206,85,49,57}},
        {"Jump", {136,35,52,182}},
        {"Jumping", {147,110,34,33}},
        {"Liquor", {256,152,73,210}},
        {"Matrix", {331,39,38,42}},
        {"MotorRolling", {117,68,122,125}},
        {"Panda", {58,100,28,23}},
        {"RedTeam", {197,87,38,18}},
        {"Shaking", {225,135,61,71}},
        {"Singer2", {298,149,67,122}},
        {"Skating1", {162,188,34,84}},
        {"Skating2", {289,67,64,236}},
        {"Skiing", {446,181,29,26}},
        {"Soccer", {302,135,67,81}},
        {"Surfer", {275,137,23,26}},
        {"Sylvester", {122,51,51,61}},
        {"Tiger2", {32,60,68,78}},
        {"Trellis", {146,54,68,101}},
        {"Walking", {692,439,24,79}},
        {"Walking2", {130,132,31,115}},
        {"Woman", {213,121,21,95}}
    };

    std::string key = "CarScale";
    std::string pattern_jpg = "./input/img_sequences/" + key + "/img/*.jpg";
    std::vector<cv::String> imgs;
    cv::glob(pattern_jpg, imgs);
    if(imgs.size() == 0)
        return -1;
    cv::Rect_<float> init_pos(map[key][0], map[key][1], map[key][2], map[key][3]);

    // string outputVideoPath = "./output/good/";
    // string outputVideoName = "Box_apce.avi";
    // cv::Mat src = cv::imread(imgs[0]);
    // cv::VideoWriter writer;
    // writer.open(outputVideoPath + outputVideoName, VideoWriter::fourcc('M', 'J', 'P', 'G'), 25.0, src.size(), src.type() == CV_8UC3);
    // if(!writer.isOpened()){
    //     std::cout << "open video writer failed." << std::endl;
    //     return -1;
    // }

    std::map<std::string, double> params{
        {"use_color_feature", 0}, 
        {"use_color_hist_model", 0},
        {"use_apce_jugment", 0},
        {"use_part_track", 0},
        {"fixed_area", 75*75} // 例如这里调整目标区域的尺寸从150*150变为75*75，速度会很大提升，精度会有所下降
    };
    // 每次使用先实例化tracker对象
    ObjTracker* tracker = new ObjTracker(params);

    cv::Mat img;
    // initialize
    img = cv::imread(imgs[0]);
    // cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);    
    tracker->InitObjTracker(img, init_pos);

    // tracking
    for(int i = 1; i < imgs.size(); i++){
        std::cout << i << "/" << imgs.size() - 1 << " ";
        img = cv::imread(imgs[i]);
        // cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        cv::Rect_<float> part_rect1, part_rect2;
        Rect bbox = tracker->TrackNextFrame(img, part_rect1, part_rect2);
        
        cv::Point_<float> part_center1(part_rect1.x + 0.5*part_rect1.width, part_rect1.y + 0.5*part_rect1.height);
        cv::Point_<float> part_center2(part_rect2.x + 0.5*part_rect2.width, part_rect2.y + 0.5*part_rect2.height);
        cv::Point_<float> rough_center(0.5*(part_center1.x + part_center2.x), 0.5*(part_center1.y + part_center2.y));
        cv::Point_<float> final_center(bbox.x + 0.5*bbox.width, bbox.y + 0.5*bbox.height);

        circle(img, part_center1, 2, Scalar(255, 0, 0), -1);
        circle(img, part_center2, 2, Scalar(255, 0, 0), -1);
        line(img, part_center1, part_center2, Scalar(0, 0, 255), 1);
        circle(img, rough_center, 2, Scalar(255, 0, 0), -1);
        rectangle(img, part_rect1, cv::Scalar(0, 255, 0), 1, 1);
        rectangle(img, part_rect2, cv::Scalar(0, 255, 0), 1, 1);
        circle(img, final_center, 2, Scalar(0, 255, 0), -1);
        rectangle(img, bbox, cv::Scalar(0, 255, 0), 1, 1);
        imshow("video", img);
        // writer.write(img);
        if(waitKey(10) == 'q')
            break;
    }

    // 不使用了需要delete释放内存空间
    delete tracker;
    return 0;
}