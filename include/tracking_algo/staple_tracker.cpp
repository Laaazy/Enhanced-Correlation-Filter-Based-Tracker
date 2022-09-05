/*
 * cv::Size(width, height)
 * cv::Point(x, y)
 * cv::Mat(height, width, channels, ... )
 * cv::Mat save by row after row
 *   2d: address = j * width + i
 *   3d: address = j * width * channels + i * channels + k
 * ------------------------------------------------------------
 * row == height == Point.y
 * col == width == Point.x
 * Mat::at(Point(x, y)) == Mat::at(y,x)
 */
#include <iostream>
#include <iomanip>
#include <fstream>
#include "fhog.h"
#include "apce.hpp"
#include "staple_tracker.hpp"

// mexResize got different results using different OpenCV, it's not trustable
// I found this bug by running vot2015/tunnel, it happened when frameno+1==22 after frameno+1==21
void STAPLE_TRACKER::mexResize(const cv::Mat &im, cv::Mat &output, cv::Size newsz, const char *method){
    int interpolation = cv::INTER_LINEAR;
    double inv_scale_x, inv_scale_y;
    inv_scale_x = (double) newsz.width / im.cols;
    inv_scale_y = (double) newsz.height / im.rows;
    cv::resize(im, output, newsz, inv_scale_x, inv_scale_y, interpolation);
}


staple_cfg STAPLE_TRACKER::default_parameters_staple(staple_cfg cfg){
    return cfg;
}


void STAPLE_TRACKER::initializeAllAreas(const cv::Mat &im){
    // initialize:
    // bg_area: target_width/height + 0.5*(target_width + target_height)
    // fg_area: target_width/height - 0.1*(target_width + target_height)
    // norm_bg_area: norm bg_area size to cfg.fixed_area
    // norm_target_sz: target_size * area_resize_factor
    // cf_response_size: norm_bg_area / hog_cell_size
    // norm_delta_area: area_resize_factor*0.5*(target_size.width+ target_size.height) for w/h, a square
    // norm_pwp_search_area: same size as norm_bg_area

    // we want a regular frame surrounding the object
    double avg_dim = (cfg.target_sz.width + cfg.target_sz.height) / 2.0;

    bg_area.width = round(cfg.target_sz.width + avg_dim);
    bg_area.height = round(cfg.target_sz.height + avg_dim);

    // pick a "safe" region smaller than bbox to avoid mislabeling
    fg_area.width = round(cfg.target_sz.width - avg_dim * cfg.inner_padding);
    fg_area.height = round(cfg.target_sz.height - avg_dim * cfg.inner_padding);

    // saturate to image size
    cv::Size imsize = im.size();

    bg_area.width = std::min(bg_area.width, imsize.width - 1);
    bg_area.height = std::min(bg_area.height, imsize.height - 1);

    // make sure the differences are a multiple of 2 (makes things easier later in color histograms)
    bg_area.width = bg_area.width - (bg_area.width - cfg.target_sz.width) % 2;
    bg_area.height = bg_area.height - (bg_area.height - cfg.target_sz.height) % 2;

    fg_area.width = fg_area.width + (bg_area.width - fg_area.width) % 2;
    fg_area.height = fg_area.height + (bg_area.height - fg_area.width) % 2;

    // Compute the rectangle with (or close to) params.fixedArea
    // and same aspect ratio as the target bbox
    area_resize_factor = sqrt(cfg.fixed_area / double(bg_area.width * bg_area.height));
    norm_bg_area.width = round(bg_area.width * area_resize_factor);
    norm_bg_area.height = round(bg_area.height * area_resize_factor);

    // Correlation Filter (HOG) feature space
    // It smaller that the norm bg area if HOG cell size is > 1
    cf_response_size.width = floor(norm_bg_area.width / cfg.hog_cell_size);
    cf_response_size.height = floor(norm_bg_area.height / cfg.hog_cell_size);

    // given the norm BG area, which is the corresponding target w and h?
    double norm_target_sz_w = 0.75*norm_bg_area.width - 0.25*norm_bg_area.height; // actually cfg.target_sz.width * area_resize_factor
    double norm_target_sz_h = 0.75*norm_bg_area.height - 0.25*norm_bg_area.width; // cfg.target_sz.height * area_resize_factor

    norm_target_sz.width = round(norm_target_sz_w);
    norm_target_sz.height = round(norm_target_sz_h);

    // distance (on one side) between target and bg area
    cv::Size norm_pad;

    norm_pad.width = floor((norm_bg_area.width - norm_target_sz.width) / 2.0);
    norm_pad.height = floor((norm_bg_area.height - norm_target_sz.height) / 2.0);

    int radius = floor(fmin(norm_pad.width, norm_pad.height));

    // norm_delta_area is the number of rectangles that are considered.
    // its actually size is 
    // (area_resize_factor*0.5*(target_size.width+ target_size.height), area_resize_factor*0.5*(target_size.width+ target_size.height))
    // it is the "sampling space" and the dimension of the final merged resposne
    // it is squared to not privilege any particular direction
    norm_delta_area = cv::Size((2*radius+1), (2*radius+1));

    // Rectangle in which the integral images are computed.
    // Grid of rectangles ( each of size norm_target_sz) has size norm_delta_area.
    // actually the same size as norm_bg_area
    norm_pwp_search_area.width = norm_target_sz.width + norm_delta_area.width - 1;
    norm_pwp_search_area.height = norm_target_sz.height + norm_delta_area.height - 1;
}


// GET_SUBWINDOW Obtain image sub-window, padding is done by replicating border values.
// Returns sub-window of image IM centered at POS ([y, x] coordinates),
// with size MODEL_SZ ([height, width]). If any pixels are outside of the image,
// they will replicate the values at the borders
void STAPLE_TRACKER::getSubwindow(const cv::Mat &im, cv::Point_<float> centerCoor, cv::Size model_sz, cv::Size scaled_sz, cv::Mat &output){
    cv::Size sz = scaled_sz; // scale adaptation
    // make sure the size is not to small
    sz.width = fmax(sz.width, 2);
    sz.height = fmax(sz.height, 2);

    cv::Mat subWindow;

    cv::Point lefttop(
        std::min(im.cols - 1, std::max(-sz.width + 1, int(centerCoor.x + 1 - sz.width/2.0+0.5))),
        std::min(im.rows - 1, std::max(-sz.height + 1, int(centerCoor.y + 1 - sz.height/2.0+0.5)))
    );
    cv::Point rightbottom(
        std::max(0, int(lefttop.x + sz.width - 1)),
        std::max(0, int(lefttop.y + sz.height - 1))
    );

    cv::Point lefttopLimit(
        std::max(lefttop.x, 0),
        std::max(lefttop.y, 0)
    );
    cv::Point rightbottomLimit(
        std::min(rightbottom.x, im.cols - 1),
        std::min(rightbottom.y, im.rows - 1)
    );

    rightbottomLimit.x += 1;
    rightbottomLimit.y += 1;
    cv::Rect roiRect(lefttopLimit, rightbottomLimit);

    // crop subWindow of size scaled_sz
    im(roiRect).copyTo(subWindow);
	
    int top = lefttopLimit.y - lefttop.y;
    int bottom = rightbottom.y - rightbottomLimit.y + 1;
    int left = lefttopLimit.x - lefttop.x;
    int right = rightbottom.x - rightbottomLimit.x + 1;
    cv::copyMakeBorder(subWindow, subWindow, top, bottom, left, right, cv::BORDER_REPLICATE);

    // resize subWindow to model_sz
    mexResize(subWindow, output, model_sz, "auto");
}


// UPDATEHISTMODEL create new models for foreground and background or update the current ones
void STAPLE_TRACKER::updateHistModel(bool new_model, cv::Mat &patch, double learning_rate_pwp){
    cv::Size pad_offset1;
    // we constrained the difference to be mod2, so we do not have to round here
    pad_offset1.width = (bg_area.width - target_sz.width) / 2;
    pad_offset1.height = (bg_area.height - target_sz.height) / 2;

    // difference between bg_area and target_sz has to be even
    if (
        (
            (pad_offset1.width == round(pad_offset1.width)) &&
            (pad_offset1.height != round(pad_offset1.height))
        ) ||
        (
            (pad_offset1.width != round(pad_offset1.width)) &&
            (pad_offset1.height == round(pad_offset1.height))
        )) {
        assert(0);
    }

    pad_offset1.width = fmax(pad_offset1.width, 1);
    pad_offset1.height = fmax(pad_offset1.height, 1);

    cv::Mat bg_mask(bg_area, CV_8UC1, cv::Scalar(1)); // init bg_mask

    // cv::Rect(x, y, w, h)
    // Rect of target
    cv::Rect pad1_rect(
        pad_offset1.width,
        pad_offset1.height,
        bg_area.width - 2 * pad_offset1.width,
        bg_area.height - 2 * pad_offset1.height
    );
    bg_mask(pad1_rect) = false;

    cv::Size pad_offset2;
    // we constrained the difference to be mod2, so we do not have to round here
    pad_offset2.width = (bg_area.width - fg_area.width) / 2;
    pad_offset2.height = (bg_area.height - fg_area.height) / 2;

    // difference between bg_area and fg_area has to be even
    if (
        (
            (pad_offset2.width == round(pad_offset2.width)) &&
            (pad_offset2.height != round(pad_offset2.height))
        ) ||
        (
            (pad_offset2.width != round(pad_offset2.width)) &&
            (pad_offset2.height == round(pad_offset2.height))
        )) {
        assert(0);
    }

    pad_offset2.width = fmax(pad_offset2.width, 1);
    pad_offset2.height = fmax(pad_offset2.height, 1);

    cv::Mat fg_mask(bg_area, CV_8UC1, cv::Scalar(0)); // init fg_mask

    // Rect of fg_area
    cv::Rect pad2_rect(
        pad_offset2.width,
        pad_offset2.height,
        bg_area.width - 2 * pad_offset2.width,
        bg_area.height - 2 * pad_offset2.height
    );
    fg_mask(pad2_rect) = true;

    cv::Mat fg_mask_new;
    cv::Mat bg_mask_new;
    mexResize(fg_mask, fg_mask_new, norm_bg_area, "auto");
    mexResize(bg_mask, bg_mask_new, norm_bg_area, "auto");

    int imgCount = 1;
    int dims = cfg.grayscale_sequence ? 1 : 3;
    const int sizes[] = { cfg.n_bins, cfg.n_bins, cfg.n_bins };
    const int channels[] = { 0, 1, 2 };
    float bRange[] = { 0, 256 };
    float gRange[] = { 0, 256 };
    float rRange[] = { 0, 256 };
    const float *ranges[] = { bRange, gRange, rRange };

    // (TRAIN) BUILD THE MODEL
    if (new_model) {
        cv::calcHist(&patch, imgCount, channels, bg_mask_new, bg_hist, dims, sizes, ranges);
        cv::calcHist(&patch, imgCount, channels, fg_mask_new, fg_hist, dims, sizes, ranges);
    } else {
        cv::MatND bg_hist_tmp;
        cv::MatND fg_hist_tmp;

        cv::calcHist(&patch, imgCount, channels, bg_mask_new, bg_hist_tmp, dims, sizes, ranges);
        cv::calcHist(&patch, imgCount, channels, fg_mask_new, fg_hist_tmp, dims, sizes, ranges);

        bg_hist = (1 - learning_rate_pwp)*bg_hist + learning_rate_pwp*bg_hist_tmp;
        fg_hist = (1 - learning_rate_pwp)*fg_hist + learning_rate_pwp*fg_hist_tmp;
    }
}


void STAPLE_TRACKER::CalculateHann(cv::Size sz, cv::Mat &output){
    cv::Mat temp1(cv::Size(sz.width, 1), CV_32FC1);
    cv::Mat temp2(cv::Size(sz.height, 1), CV_32FC1);

    float *p1 = temp1.ptr<float>(0);
    float *p2 = temp2.ptr<float>(0);

    for (int i = 0; i < sz.width; ++i)
        p1[i] = 0.5*(1 - cos(CV_2PI*i / (sz.width - 1)));

    for (int i = 0; i < sz.height; ++i)
        p2[i] = 0.5*(1 - cos(CV_2PI*i / (sz.height - 1)));

    output = temp2.t()*temp1;
}


void meshgrid(const cv::Range xr, const cv::Range yr, cv::Mat &outX, cv::Mat &outY){
    std::vector<int> x, y;

    for (int i = xr.start; i <= xr.end; i++)
        x.push_back(i);
    for (int i = yr.start; i <= yr.end; i++)
        y.push_back(i);

    repeat(cv::Mat(x).t(), y.size(), 1, outX);
    repeat(cv::Mat(y), 1, x.size(), outY);
}


// GAUSSIANRESPONSE create the (fixed) target response of the correlation filter response
void STAPLE_TRACKER::gaussianResponse(cv::Size rect_size, double sigma, cv::Mat &output){
    cv::Size half;

    half.width = floor((rect_size.width - 1) / 2);
    half.height = floor((rect_size.height - 1) / 2);

    cv::Range i_range(-half.width, rect_size.width - (1 + half.width));
    cv::Range j_range(-half.height, rect_size.height - (1 + half.height));
    cv::Mat i, j;

    meshgrid(i_range, j_range, i, j);

    std::vector<int> i_mod_range;
    i_mod_range.reserve(i_range.end - i_range.start + 1);
    std::vector<int> j_mod_range;
    i_mod_range.reserve(j_range.end - j_range.start + 1);

    for (int k = i_range.start; k <= i_range.end; k++) {
        int val = (int)(k - 1 + rect_size.width) % (int)rect_size.width;
        i_mod_range.push_back(val);
    }

    for (int k = j_range.start; k <= j_range.end; k++) {
        int val = (int)(k - 1 + rect_size.height) % (int)rect_size.height;
        j_mod_range.push_back(val);
    }

    output = cv::Mat(rect_size.height, rect_size.width, CV_32FC2);

    for (int jj = 0; jj < rect_size.height; jj++){
        int j_idx = j_mod_range[jj];
        assert(j_idx < rect_size.height);

        for (int ii = 0; ii < rect_size.width; ii++){
            int i_idx = i_mod_range[ii];
            assert(i_idx < rect_size.width);

            cv::Vec2f val(exp(-(i.at<int>(jj, ii)*i.at<int>(jj, ii) + j.at<int>(jj, ii)*j.at<int>(jj, ii)) / (2 * sigma*sigma)), 0);
            output.at<cv::Vec2f>(j_idx, i_idx) = val;
        }
    }
}


// load w2c
void STAPLE_TRACKER::load_w2c(){
    ifstream ifstr("./w2crs.txt");
    w2c = vector<vector<double>>(10, vector<double>(32768, 0));
    double tmp_val;
    for(int i = 0; i < 10; i++){
        for(int j = 0; j < 32768; j++){
            ifstr >> tmp_val;
            w2c[i][j] = tmp_val;
        }
    }
    ifstr.close();
}


// part initialize
void STAPLE_TRACKER::part_initialize(cv::Mat im, cv::Rect_<float> region){
    // part rect
    if(region.height > region.width){
        is_tall = true;
        cv::Rect_<float> rect_part1(region.x, region.y, region.width, region.height / 2);
        cv::Rect_<float> rect_part2(region.x, region.y + region.height / 2, region.width, region.height / 2);
        rect_part.push_back(rect_part1);
        rect_part.push_back(rect_part2);
    }
    else{
        is_tall = false;
        cv::Rect_<float> rect_part1(region.x, region.y, region.width / 2, region.height);
        cv::Rect_<float> rect_part2(region.x + region.width / 2, region.y, region.width / 2, region.height);
        rect_part.push_back(rect_part1);
        rect_part.push_back(rect_part2);
    }
    
    // size of part response    
    cv::Size part_sz(rect_part[0].width, rect_part[0].height);
    double avg_dim_part = (part_sz.width + part_sz.height) / 2.0;

    bg_area_part.width = round(part_sz.width + avg_dim_part);
    bg_area_part.height = round(part_sz.height + avg_dim_part);

    bg_area_part.width = std::min(bg_area_part.width, im.size().width - 1);
    bg_area_part.height = std::min(bg_area_part.height, im.size().height - 1);

    bg_area_part.width = bg_area_part.width - (bg_area_part.width - part_sz.width) % 2;
    bg_area_part.height = bg_area_part.height - (bg_area_part.height - part_sz.height) % 2;

    area_resize_factor_part = sqrt(cfg.fixed_area_part / double(bg_area_part.width * bg_area_part.height));
    norm_bg_area_part.width = round(bg_area_part.width * area_resize_factor_part);
    norm_bg_area_part.height = round(bg_area_part.height * area_resize_factor_part);

    cf_response_size_part.width = floor(norm_bg_area_part.width / cfg.hog_cell_size);
    cf_response_size_part.height = floor(norm_bg_area_part.height / cfg.hog_cell_size);

    norm_target_sz_part.width = round(0.75*norm_bg_area_part.width - 0.25*norm_bg_area_part.height);
    norm_target_sz_part.height = round(0.75*norm_bg_area_part.height - 0.25*norm_bg_area_part.width);

    cv::Size norm_pad_part;
    norm_pad_part.width = floor((norm_bg_area_part.width - norm_target_sz_part.width) / 2.0);
    norm_pad_part.height = floor((norm_bg_area_part.height - norm_target_sz_part.height) / 2.0);

    int radius = floor(fmin(norm_pad_part.width, norm_pad_part.height));
    norm_delta_area_part = cv::Size((2*radius+1), (2*radius+1));
    
    // part hann window
    CalculateHann(cf_response_size_part, hann_window_part); 
    
    // part gaussian label
    double output_sigma_part = sqrt(norm_target_sz_part.width * norm_target_sz_part.height) * cfg.output_sigma_factor / cfg.hog_cell_size;
    cv::Mat y_part;
    gaussianResponse(cf_response_size_part, output_sigma_part, y_part);
    cv::dft(y_part, yf_part);

    // valid part
    valid_part.push_back(true);
    valid_part.push_back(true);

    // part center distance
    cv::Point_<float> center_part1(rect_part[0].x + rect_part[0].width / 2.0, rect_part[0].y + rect_part[0].height / 2.0);
    cv::Point_<float> center_part2(rect_part[1].x + rect_part[1].width / 2.0, rect_part[1].y + rect_part[1].height / 2.0);
    init_part_dist = sqrt(pow(center_part1.x - center_part2.x, 2) + pow(center_part1.y - center_part2.y, 2));
}


void STAPLE_TRACKER::tracker_staple_initialize(cv::Mat im, cv::Rect_<float> region){
    int n = im.channels();
    if (n == 1) 
        cfg.grayscale_sequence = true;

    cfg.init_pos.x = region.x + region.width / 2.0;
    cfg.init_pos.y = region.y + region.height / 2.0;

    double w = region.width;
    double h = region.height;
    cfg.target_sz.width = round(w);
    cfg.target_sz.height = round(h);
    // if(sqrt(cfg.target_sz.height * cfg.target_sz.width) < 38)
    //     cfg.hog_cell_size = 2;

    initializeAllAreas(im);

    pos = cfg.init_pos;
    target_sz = cfg.target_sz;

    cv::Mat patch_padded;
    getSubwindow(im, pos, norm_bg_area, bg_area, patch_padded);

    // initialize hist model
    if(cfg.use_color_hist_model)
        updateHistModel(true, patch_padded);

    CalculateHann(cf_response_size, hann_window);

    // gaussian-shaped desired response, centred in (1,1)
    // bandwidth proportional to target size
    double output_sigma = sqrt(norm_target_sz.width * norm_target_sz.height) * cfg.output_sigma_factor / cfg.hog_cell_size;
    cv::Mat y;
    gaussianResponse(cf_response_size, output_sigma, y);
    cv::dft(y, yf);

    // SCALE ADAPTATION INITIALIZATION
    if (cfg.scale_adaptation){
        // Code from DSST
        scale_factor = 1;
        base_target_sz = target_sz;
        float scale_sigma = sqrt(cfg.num_scales) * cfg.scale_sigma_factor;

        cv::Mat ys = cv::Mat(1, cfg.num_scales, CV_32FC2);
        for (int i = 0; i < cfg.num_scales; i++){
            cv::Vec2f val((i + 1) - ceil(cfg.num_scales/2.0f), 0.f);
            val[0] = exp(-0.5 * (val[0] * val[0]) / (scale_sigma * scale_sigma));
            ys.at<cv::Vec2f>(i) = val;
        }

        cv::dft(ys, ysf, cv::DFT_ROWS);

        scale_window = cv::Mat(1, cfg.num_scales, CV_32FC1);
        if (cfg.num_scales % 2 == 0){
            for (int i = 0; i < cfg.num_scales + 1; ++i){
                if (i > 0) 
                    scale_window.at<float>(i - 1) = 0.5*(1 - cos(CV_2PI*i / (cfg.num_scales + 1 - 1)));
            }
        }
        else{
            for (int i = 0; i < cfg.num_scales; ++i)
                scale_window.at<float>(i) = 0.5*(1 - cos(CV_2PI*i / (cfg.num_scales - 1)));
        }

        scale_factors = cv::Mat(1, cfg.num_scales, CV_32FC1);
        for (int i = 0; i < cfg.num_scales; i++)
            scale_factors.at<float>(i) = pow(cfg.scale_step, (ceil(cfg.num_scales/2.0)  - (i+1)));
        // std::cout << scale_factors << std::endl;

        if ((cfg.scale_model_factor*cfg.scale_model_factor) * (norm_target_sz.width*norm_target_sz.height) > cfg.scale_model_max_area)
            cfg.scale_model_factor = sqrt(cfg.scale_model_max_area/(norm_target_sz.width*norm_target_sz.height));

        scale_model_sz.width = floor(norm_target_sz.width * cfg.scale_model_factor);
        scale_model_sz.height = floor(norm_target_sz.height * cfg.scale_model_factor);

        // find maximum and minimum scales
        // min_scale_factor = pow(cfg.scale_step, ceil(log(std::max(5.0/bg_area.width, 5.0/bg_area.height))/log(cfg.scale_step)));
        // max_scale_factor = pow(cfg.scale_step, floor(log(std::min(im.cols/(float)target_sz.width, im.rows/(float)target_sz.height))/log(cfg.scale_step)));
        // 0.0338353 2.48661
        // 0.0206237 4.68612
        // std::cout << min_scale_factor << " " << max_scale_factor << std::endl;
    }
    if(cfg.use_color_feature)
        load_w2c();
    if(cfg.use_part_track)
        part_initialize(im, region);
}


void STAPLE_TRACKER::getColorNameFeat(const cv::Mat &im, cv::Mat &output){
    Mat double_im;
    im.convertTo(double_im, CV_64FC1);
    // cal index_im for each pixel location in im
    vector<Mat> im_split;
	cv::split(double_im, im_split);
	Mat RR = im_split[2];
	Mat GG = im_split[1];
	Mat BB = im_split[0];
	
	double*  RRdata = ((double*)RR.data), *GGdata = ((double*)GG.data), *BBdata = ((double*)BB.data);
	int w = RR.cols;
	int h = RR.rows;
	vector<int> index_im(w * h, 0);

	for (int i = 0; i < index_im.size(); i++)
		index_im[i] = (int)(floor(RRdata[i] / 8) + 32 * floor(GGdata[i] / 8) + 32 * 32 * floor(BBdata[i] / 8));
    
    // output colorname feat
    vector<Mat> out;
    for(int i = 0; i < 10; i++){
        vector<double> selected(index_im.size(), 0);
        for(int k = 0; k < index_im.size(); k++){
            selected[i] = w2c[i][index_im[k]];
        }

        Mat tmp(h, w, CV_64FC1);
        double* data = ((double*)tmp.data);
	    memcpy(data, ((double*)(&selected[0])), h*w*sizeof(double));
        out.push_back(tmp);
    }
    cv::merge(out, output);
    #undef CHANNELS
    #define CHANNELS 10
    output.convertTo(output, CV_32FC(CHANNELS));
}


// code from DSST
void STAPLE_TRACKER::getFeatureMap(cv::Mat &im_patch, const char *feature_type, cv::MatND &output){
    assert(!strcmp(feature_type, "fhog"));

    cv::Mat hog_feat;
    fhog28(hog_feat, im_patch, cfg.hog_cell_size, 9);

    int w = hog_feat.cols;
    int h = hog_feat.rows;

    cv::Mat new_im_patch;
    if (cfg.hog_cell_size > 1) {
        cv::Size newsz(w, h);
        mexResize(im_patch, new_im_patch, newsz, "auto");
    } else 
        new_im_patch = im_patch;

    output = hog_feat;

    if(cfg.use_color_feature){
        cv::Mat color_name_feat;
        getColorNameFeat(new_im_patch, color_name_feat);
        
        // merge hog and colorName feat
        vector<cv::Mat> tmp;
        tmp.push_back(hog_feat.clone());
        tmp.push_back(color_name_feat.clone());
        cv::merge(tmp, output);
    }

    cv::Mat grayimg;
    if (new_im_patch.channels() > 1)
        cv::cvtColor(new_im_patch, grayimg, cv::COLOR_BGR2GRAY);
    else 
        grayimg = new_im_patch;

    float alpha = 1. / 255.0;
    float betta = 0.5;

    // apply Hann window
    if(cfg.use_color_feature){
        typedef cv::Vec<float, 38> Vecf38;
        for (int j = 0; j < h; ++j){
            Vecf38* pDst = output.ptr<Vecf38>(j);
            const float* pHann = hann_window.ptr<float>(j);
            const uchar* pGray = grayimg.ptr<uchar>(j);
            for (int i = 0; i < w; ++i){
                Vecf38& val = pDst[i];
                val *= pHann[i];
                val[0] = (alpha * pGray[i] - betta) * pHann[i];
            }
        }
    }
    else{
        typedef cv::Vec<float, 28> Vecf28;
        for (int j = 0; j < h; ++j){
            Vecf28* pDst = output.ptr<Vecf28>(j);
            const float* pHann = hann_window.ptr<float>(j);
            const uchar* pGray = grayimg.ptr<uchar>(j);
            for (int i = 0; i < w; ++i){
                Vecf28& val = pDst[i];
                val *= pHann[i];
                val[0] = (alpha * pGray[i] - betta) * pHann[i];
            }
        }
    }
}


void matsplit(const cv::MatND &xt, std::vector<cv::Mat> &xtsplit){
    int w = xt.cols;
    int h = xt.rows;
    int cn = xt.channels();

    // assert(cn == 28);

    for (int k = 0; k < cn; k++){
        cv::Mat dim = cv::Mat(h, w, CV_32FC2);
        for (int j = 0; j < h; ++j){
            float* pDst = dim.ptr<float>(j);
            const float* pSrc = xt.ptr<float>(j);
            for (int i = 0; i < w; ++i){
                pDst[0] = pSrc[k];
                pDst[1] = 0.0f;

                pSrc += cn;
                pDst += 2;
            }
        }
        xtsplit.push_back(dim);
    }
}


// GET_SUBWINDOW Obtain image sub-window, padding is done by replicating border values.
// Returns sub-window of image IM centered at POS ([y, x] coordinates),
// with size MODEL_SZ ([height, width]). If any pixels are outside of the image,
// they will replicate the values at the borders
void STAPLE_TRACKER::getSubwindowFloor(const cv::Mat &im, cv::Point_<float> centerCoor, cv::Size model_sz, cv::Size scaled_sz, cv::Mat &output){
    cv::Size sz = scaled_sz; // scale adaptation

    // make sure the size is not to small
    sz.width = fmax(sz.width, 2);
    sz.height = fmax(sz.height, 2);

    cv::Point lefttop(
        std::min(im.cols - 1, std::max(-sz.width + 1, int(centerCoor.x + 1) - int(sz.width/2.0))),
        std::min(im.rows - 1, std::max(-sz.height + 1, int(centerCoor.y + 1) - int(sz.height/2.0)))
    );
    cv::Point rightbottom(
        std::max(0, int(lefttop.x + sz.width - 1)),
        std::max(0, int(lefttop.y + sz.height - 1))
    );

    cv::Point lefttopLimit(
        std::max(lefttop.x, 0),
        std::max(lefttop.y, 0)
    );
    cv::Point rightbottomLimit(
        std::min(rightbottom.x, im.cols - 1),
        std::min(rightbottom.y, im.rows - 1)
    );

    rightbottomLimit.x += 1;
    rightbottomLimit.y += 1;
    cv::Rect roiRect(lefttopLimit, rightbottomLimit);
    cv::Mat subWindow;

    im(roiRect).copyTo(subWindow);

    mexResize(subWindow, output, model_sz, "auto");
}


// code from DSST
void STAPLE_TRACKER::getScaleSubwindow(const cv::Mat &im, cv::Point_<float> centerCoor, cv::Mat &output){
    int ch = 0;
    int total = 0;

    for (int s = 0; s < cfg.num_scales; s++){
        cv::Size_<float> patch_sz;

        patch_sz.width = floor(base_target_sz.width * scale_factor * scale_factors.at<float>(s));
        patch_sz.height = floor(base_target_sz.height * scale_factor * scale_factors.at<float>(s));

        cv::Mat im_patch_resized;
        getSubwindowFloor(im, centerCoor, scale_model_sz, patch_sz, im_patch_resized);

        // extract scale features
        cv::Mat temp;
        fhog31(temp, im_patch_resized, cfg.hog_cell_size, 9);

        // colorName 
        if(cfg.use_color_feature){
            int w = temp.cols;
            int h = temp.rows;

            cv::Mat new_im_patch;
            if (cfg.hog_cell_size > 1) {
                cv::Size newsz(w, h);
                mexResize(im_patch_resized, new_im_patch, newsz, "auto");
            } else 
                new_im_patch = im_patch_resized;

            cv::Mat color_name_feat;
            getColorNameFeat(new_im_patch, color_name_feat);

            // merge hog and colorName feat
            vector<cv::Mat> tmp;
            tmp.push_back(temp.clone());
            tmp.push_back(color_name_feat.clone());
            cv::merge(tmp, temp);
        }
    
        if (s == 0){
            ch = temp.channels();
            total = temp.cols * temp.rows * ch;
            output = cv::Mat(total, cfg.num_scales, CV_32FC2);
        }

        int tempw = temp.cols;
        int temph = temp.rows;
        int tempch = temp.channels();

        int count = 0;

        float scaleWnd = scale_window.at<float>(s);
        float* outData = (float*)output.data;
        // window
        for (int j = 0; j < temph; ++j){
            const float* tmpData = temp.ptr<float>(j);
            for (int i = 0; i < tempw; ++i){
                for (int k = 0; k < tempch; ++k){
                    outData[(count * cfg.num_scales + s) * 2 + 0] = tmpData[k] * scaleWnd;
                    outData[(count * cfg.num_scales + s) * 2 + 1] = 0.0;
                    count++;
                }
                tmpData += ch;
            }
        }
    }
}


// part train
void STAPLE_TRACKER::part_train(cv::Mat im, bool is_first){
    // part hf_den and hf_num
    for(int p = 0; p < rect_part.size(); p++){
        if(!is_first && !valid_part[p])
            continue;

        cv::Point2f center_part(rect_part[p].x + rect_part[p].width / 2.0, rect_part[p].y + rect_part[p].height / 2.0);
        cv::Mat im_patch_bg_part;

        getSubwindow(im, center_part, norm_bg_area_part, bg_area_part, im_patch_bg_part);

        cv::MatND xt_part;
        getFeatureMap(im_patch_bg_part, cfg.feature_type, xt_part);

        std::vector<cv::Mat> xtsplit_part;
        matsplit(xt_part, xtsplit_part);

        std::vector<cv::Mat> xtf_part;
        for (int i =  0; i < xt_part.channels(); i++) {
            cv::Mat dimf;
            cv::dft(xtsplit_part[i], dimf);
            xtf_part.push_back(dimf);
        }

        std::vector<cv::Mat> new_hf_num_part;
        std::vector<cv::Mat> new_hf_den_part;

        int w = xt_part.cols;
        int h = xt_part.rows;
        float invArea = 1.f / (cf_response_size_part.width * cf_response_size_part.height);

        // G*F
        for (int ch = 0; ch < xt_part.channels(); ch++){
            cv::Mat dim = cv::Mat(h, w, CV_32FC2);
            for (int j = 0; j < h; ++j){
                const float* pXTF = xtf_part[ch].ptr<float>(j);
                const float* pYF = yf_part.ptr<float>(j);
                cv::Vec2f* pDst = dim.ptr<cv::Vec2f>(j);

                for (int i = 0; i < w; ++i){
                    cv::Vec2f val(pYF[1] * pXTF[1] + pYF[0] * pXTF[0], pYF[0] * pXTF[1] - pYF[1] * pXTF[0]); // actually(pYF[0] * pXTF[0], 0)
                    *pDst = invArea * val;

                    pXTF += 2;
                    pYF += 2;
                    ++pDst;
                }
            }
            new_hf_num_part.push_back(dim);
        }

        // F*F
        for (int ch = 0; ch < xt_part.channels(); ch++){
            cv::Mat dim = cv::Mat(h, w, CV_32FC1);
            for (int j = 0; j < h; ++j){
                const float* pXTF = xtf_part[ch].ptr<float>(j);
                float* pDst = dim.ptr<float>(j);

                for (int i = 0; i < w; ++i){
                    *pDst = invArea * (pXTF[0]*pXTF[0] + pXTF[1]*pXTF[1]);

                    pXTF += 2;
                    ++pDst;
                }
            }
            new_hf_den_part.push_back(dim);
        }

        if(is_first){
            hf_num_part.push_back(new_hf_num_part);
            hf_den_part.push_back(new_hf_den_part);
        }
        else{
            for (int ch =  0; ch < xt_part.channels(); ch++) {
                hf_num_part[p][ch] = (1 - cfg.learning_rate_cf_part) * hf_num_part[p][ch] + cfg.learning_rate_cf_part * new_hf_num_part[ch];
                hf_den_part[p][ch] = (1 - cfg.learning_rate_cf_part) * hf_den_part[p][ch] + cfg.learning_rate_cf_part * new_hf_den_part[ch];
            }
        }
    }
}


// TRAINING
void STAPLE_TRACKER::tracker_staple_train(cv::Mat im, bool first){
    if(cfg.use_apce_jugment && !apce_update)
        return;
    // extract patch of size bg_area and resize to norm_bg_area
    cv::Mat im_patch_bg;
    getSubwindow(im, pos, norm_bg_area, bg_area, im_patch_bg);

    // compute feature map, of cf_response_size
    cv::MatND xt;
    getFeatureMap(im_patch_bg, cfg.feature_type, xt);

    // compute FFT
    // cv::MatND xtf;
    std::vector<cv::Mat> xtsplit;
    std::vector<cv::Mat> xtf; // xtf is splits of xtf

    matsplit(xt, xtsplit);

    for (int i =  0; i < xt.channels(); i++) {
        cv::Mat dimf;
        cv::dft(xtsplit[i], dimf);
        xtf.push_back(dimf);
    }

    // FILTER UPDATE
    // Compute expectations over circular shifts,
    // therefore divide by number of pixels.
    {
        std::vector<cv::Mat> new_hf_num;
        std::vector<cv::Mat> new_hf_den;

        int w = xt.cols;
        int h = xt.rows;
        float invArea = 1.f / (cf_response_size.width * cf_response_size.height);

        // G*F
        for (int ch = 0; ch < xt.channels(); ch++){
            cv::Mat dim = cv::Mat(h, w, CV_32FC2);
            for (int j = 0; j < h; ++j){
                const float* pXTF = xtf[ch].ptr<float>(j);
                const float* pYF = yf.ptr<float>(j);
                cv::Vec2f* pDst = dim.ptr<cv::Vec2f>(j);

                for (int i = 0; i < w; ++i){
                    cv::Vec2f val(pYF[1] * pXTF[1] + pYF[0] * pXTF[0], pYF[0] * pXTF[1] - pYF[1] * pXTF[0]); // actually(pYF[0] * pXTF[0], 0)
                    *pDst = invArea * val;

                    pXTF += 2;
                    pYF += 2;
                    ++pDst;
                }
            }
            new_hf_num.push_back(dim);
        }

        // F*F
        for (int ch = 0; ch < xt.channels(); ch++){
            cv::Mat dim = cv::Mat(h, w, CV_32FC1);
            for (int j = 0; j < h; ++j){
                const float* pXTF = xtf[ch].ptr<float>(j);
                float* pDst = dim.ptr<float>(j);

                for (int i = 0; i < w; ++i){
                    *pDst = invArea * (pXTF[0]*pXTF[0] + pXTF[1]*pXTF[1]);

                    pXTF += 2;
                    ++pDst;
                }
            }
            new_hf_den.push_back(dim);
        }

        if (first) {
            hf_den.assign(new_hf_den.begin(), new_hf_den.end());
            hf_num.assign(new_hf_num.begin(), new_hf_num.end());
        } 
        else {
            for (int ch =  0; ch < xt.channels(); ch++) {
                hf_den[ch] = (1 - cfg.learning_rate_cf) * hf_den[ch] + cfg.learning_rate_cf * new_hf_den[ch];
                hf_num[ch] = (1 - cfg.learning_rate_cf) * hf_num[ch] + cfg.learning_rate_cf * new_hf_num[ch];
            }
            if(cfg.use_color_hist_model){
                // BG/FG COLOR MODEL UPDATE
                updateHistModel(false, im_patch_bg, cfg.learning_rate_pwp);
            }
        }
    }

    // SCALE UPDATE
    if (cfg.scale_adaptation) {
        cv::Mat im_patch_scale;
        getScaleSubwindow(im, pos, im_patch_scale);

        cv::Mat xsf;
        cv::dft(im_patch_scale, xsf, cv::DFT_ROWS);

        cv::Mat new_sf_num;
        cv::Mat new_sf_den;

        int w = xsf.cols;
        int h = xsf.rows;

        new_sf_num = cv::Mat(h, w, CV_32FC2); // G*F
        for (int j = 0; j < h; ++j){
            float* pDst = new_sf_num.ptr<float>(j);

            const float* pXSF = xsf.ptr<float>(j);
            const float* pYSF = ysf.ptr<float>(0);

            for (int i = 0; i < w; ++i){
                pDst[0] = (pYSF[1] * pXSF[1] + pYSF[0] * pXSF[0]);
                pDst[1] = (pYSF[1] * pXSF[0] - pYSF[0] * pXSF[1]);

                pXSF += 2;
                pYSF += 2;
                pDst += 2;
            }
        }

        new_sf_den = cv::Mat(1, w, CV_32FC1, cv::Scalar(0, 0, 0));
        float* pDst = new_sf_den.ptr<float>(0);
        for (int j = 0; j < h; ++j){
            const float* pSrc = xsf.ptr<float>(j);
            for (int i = 0; i < w; ++i){
                pDst[i] += (pSrc[0] * pSrc[0] + pSrc[1] * pSrc[1]);
                pSrc += 2;
            }
        }

        if (first) {
            new_sf_den.copyTo(sf_den);
            new_sf_num.copyTo(sf_num);
        } else {
            sf_den = (1 - cfg.learning_rate_scale) * sf_den + cfg.learning_rate_scale * new_sf_den;
            sf_num = (1 - cfg.learning_rate_scale) * sf_num + cfg.learning_rate_scale * new_sf_num;
        }
    }

    // update bbox position
    if (first) {
        rect_position.x = pos.x - target_sz.width/2.0;
        rect_position.y = pos.y - target_sz.height/2.0;
        rect_position.width = target_sz.width;
        rect_position.height = target_sz.height;
    }

    frameno += 1;

    if(cfg.use_part_track)
        part_train(im, first);
}


cv::Mat ensure_real(const cv::Mat &complex){
    int w = complex.cols;
    int h = complex.rows;

    cv::Mat real = cv::Mat(h, w, CV_32FC1);
    for (int j = 0; j < h; ++j){
        float* pDst = real.ptr<float>(j);
        const float* pSrc = complex.ptr<float>(j);
        for (int i = 0; i < w; ++i){
            *pDst = *pSrc;
            ++pDst;
            pSrc += 2;
        }
    }

    return real;
}


void STAPLE_TRACKER::cropFilterResponse(const cv::Mat &response_cf, cv::Size response_size, cv::Mat& output){
    int w = response_cf.cols;
    int h = response_cf.rows;

    // newh and neww must be odd, as we want an exact center
    assert(((response_size.width % 2) == 1) && ((response_size.height % 2) == 1));

    int half_width = response_size.width / 2;
    int half_height = response_size.height / 2;

    cv::Range i_range(-half_width, response_size.width - (1 + half_width));
    cv::Range j_range(-half_height, response_size.height - (1 + half_height));

    std::vector<int> i_mod_range;
    i_mod_range.reserve(i_range.end - i_range.start + 1);
    std::vector<int> j_mod_range;
    i_mod_range.reserve(j_range.end - j_range.start + 1);

    // 38 39 40 41 42 43 44 45 46 47 0 1 2 3 4 5 6 7 8
    for (int k = i_range.start; k <= i_range.end; k++) {
        int val = (k - 1 + w) % w;
        i_mod_range.push_back(val); 
    }

    // 19 20 21 22 23 24 25 26 27 28 0 1 2 3 4 5 6 7 8
    for (int k = j_range.start; k <= j_range.end; k++) {
        int val = (k - 1 + h) % h;
        j_mod_range.push_back(val);
    }

    cv::Mat tmp = cv::Mat(response_size.height, response_size.width, CV_32FC1, cv::Scalar(0, 0, 0));
    for (int j = 0; j < response_size.height; j++){
        int j_idx = j_mod_range[j];
        assert(j_idx < h);
        float* pDst = tmp.ptr<float>(j);
        const float* pSrc = response_cf.ptr<float>(j_idx);

        for (int i = 0; i < response_size.width; i++){
            int i_idx = i_mod_range[i];
            assert(i_idx < w);
            *pDst = pSrc[i_idx];
            ++pDst;
        }
    }

    output = tmp;
}


// GETCOLOURMAP computes pixel-wise probabilities (PwP) given PATCH and models BG_HIST and FG_HIST
void STAPLE_TRACKER::getColourMap(const cv::Mat &patch, cv::Mat& output){
    // check whether the patch has 3 channels
    int h = patch.rows;
    int w = patch.cols;
    int d = patch.channels();

    // figure out which bin each pixel falls into
    int bin_width = 256 / cfg.n_bins;

    output = cv::Mat(h, w, CV_32FC1);
    if (!cfg.grayscale_sequence){
        for (int j = 0; j < h; ++j){
            const uchar* pSrc = patch.ptr<uchar>(j);
            float* pDst = output.ptr<float>(j);

            for (int i = 0; i < w; ++i){
                int b1 = pSrc[0] / bin_width;
                int b2 = pSrc[1] / bin_width;
                int b3 = pSrc[2] / bin_width;

                float* histd = (float*)bg_hist.data;
                float probg = histd[b1*cfg.n_bins*cfg.n_bins + b2*cfg.n_bins + b3];

                histd = (float*)fg_hist.data;
                float profg = histd[b1*cfg.n_bins*cfg.n_bins + b2*cfg.n_bins + b3];

                *pDst = profg / (profg + probg);

                if(isnan(*pDst))
                    *pDst = 0.0;

                pSrc += d;
                ++pDst;

                // (TODO) in theory it should be at 0.5 (unseen colors shoud have max entropy)
                // likelihood_map(isnan(likelihood_map)) = 0;
            }
        }
    }
    else{
        for (int j = 0; j < h; j++){
            const uchar* pSrc = patch.ptr<uchar>(j);
            float* pDst = output.ptr<float>(j);

            for (int i = 0; i < w; i++){
                int b = *pSrc;

                float* histd = (float*)bg_hist.data;
                float probg = histd[b];

                histd = (float*)fg_hist.data;
                float profg = histd[b];

                *pDst = profg / (profg + probg);

                if(isnan(*pDst))
                    *pDst = 0.0;

                pSrc += d;
                ++pDst;

                // (TODO) in theory it should be at 0.5 (unseen colors shoud have max entropy)
                //likelihood_map(isnan(likelihood_map)) = 0;
            }
        }
    }
}


// GETCENTERLIKELIHOOD computes the sum over rectangles of size M.
void STAPLE_TRACKER::getCenterLikelihood(const cv::Mat &object_likelihood, cv::Size m, cv::Mat& center_likelihood){
    // CENTER_LIKELIHOOD is the 'colour response'
    int h = object_likelihood.rows;
    int w = object_likelihood.cols;
    int n1 = w - m.width + 1;
    int n2 = h - m.height + 1;
    float invArea = 1.f / (m.width * m.height);

    cv::Mat temp;
    // integral images
    cv::integral(object_likelihood, temp);

    center_likelihood = cv::Mat(n2, n1, CV_32FC1);

    for (int j = 0; j < n2; ++j){
        float* pLike = reinterpret_cast<float*>(center_likelihood.ptr(j));
        for (int i = 0; i < n1; ++i){
            *pLike = invArea * (temp.at<double>(j, i) + temp.at<double>(j+m.height, i+m.width) - temp.at<double>(j, i+m.width) - temp.at<double>(j+m.height, i));
            ++pLike;
        }
    }
}


void STAPLE_TRACKER::mergeResponses(const cv::Mat &response_cf, const cv::Mat &response_pwp, cv::Mat &response){
    double alpha = cfg.merge_factor_cf;
    // MERGERESPONSES interpolates the two responses with the hyperparameter ALPHA
    response = (1 - alpha) * response_cf + alpha * response_pwp;
}


// part detect
void STAPLE_TRACKER::part_update(cv::Mat im, cv::Rect_<float> &part_rect1, cv::Rect_<float> &part_rect2){
    // part center detect
    for(int p = 0; p < rect_part.size(); p++){
        cv::Point_<float> center_part(rect_part[p].x + rect_part[p].width / 2.0, rect_part[p].y + rect_part[p].height / 2.0);
        
        cv::Mat im_patch_cf_part;
        getSubwindow(im, center_part, norm_bg_area_part, bg_area_part, im_patch_cf_part);

        cv::MatND xt_windowed_part;
        getFeatureMap(im_patch_cf_part, cfg.feature_type, xt_windowed_part);

        std::vector<cv::Mat> xtsplit_part;
        matsplit(xt_windowed_part, xtsplit_part);

        std::vector<cv::Mat> xtf_part;
        for (int i =  0; i < xt_windowed_part.channels(); i++) {
            cv::Mat dimf;
            cv::dft(xtsplit_part[i], dimf);
            xtf_part.push_back(dimf);
        }

        std::vector<cv::Mat> hf_part;
        const int w = xt_windowed_part.cols;
        const int h = xt_windowed_part.rows;
        std::vector<float> DIM1(w * h, cfg.lambda);

        // F*F
        for (int ch = 0; ch < xt_windowed_part.channels(); ++ch){
            float* pDim1 = &DIM1[0];
            for (int j = 0; j < h; ++j){
                const float* pDen = hf_den_part[p][ch].ptr<float>(j);
                for (int i = 0; i < w; ++i){
                    *pDim1 += pDen[i];
                    ++pDim1;
                }
            }
        }

        // (G*F) / (F*F)
        for (int ch = 0; ch < xt_windowed_part.channels(); ++ch){
            cv::Mat dim = cv::Mat(h, w, CV_32FC2);
            const float* pDim1 = &DIM1[0];
            for (int j = 0; j < h; ++j){
                const cv::Vec2f* pSrc = hf_num_part[p][ch].ptr<cv::Vec2f>(j);
                cv::Vec2f* pDst = dim.ptr<cv::Vec2f>(j);
                for (int i = 0; i < w; ++i){
                    *pDst = *pSrc / *pDim1;
                    ++pDim1;
                    ++pDst;
                    ++pSrc;
                }
            }
            hf_part.push_back(dim);
        }

        // Z*(G*F) / (F*F)
        cv::Mat response_cff_part = cv::Mat(h, w, CV_32FC2);
        for (int j = 0; j < h; j++){
            cv::Vec2f* pDst = response_cff_part.ptr<cv::Vec2f>(j);
            for (int i = 0; i < w; i++){
                float sum = 0.0;
                float sumi = 0.0;
                for (size_t ch = 0; ch < hf_part.size(); ch++){
                    cv::Vec2f pHF = hf_part[ch].at<cv::Vec2f>(j,i);
                    cv::Vec2f pXTF = xtf_part[ch].at<cv::Vec2f>(j,i);

                    sum += (pHF[0] * pXTF[0] + pHF[1] * pXTF[1]);
                    sumi += (pHF[0] * pXTF[1] - pHF[1] * pXTF[0]);
                }
                *pDst = cv::Vec2f(sum, sumi);
                ++pDst;
            }
        }

        cv::Mat response_cfi_part;
        cv::dft(response_cff_part, response_cfi_part, cv::DFT_SCALE|cv::DFT_INVERSE);
        cv::Mat response_cf_part = ensure_real(response_cfi_part);

        cv::Size newsz = norm_delta_area_part;
        newsz.width = floor(newsz.width / cfg.hog_cell_size);
        newsz.height = floor(newsz.height / cfg.hog_cell_size);
        (newsz.width % 2 == 0) && (newsz.width -= 1);
        (newsz.height % 2 == 0) && (newsz.height -= 1);

        cropFilterResponse(response_cf_part, newsz, response_cf_part);
        if (cfg.hog_cell_size > 1){
            cv::Mat temp;
            mexResize(response_cf_part, temp, norm_delta_area_part, "auto");
            response_cf_part = temp;
        }

        double maxVal = 0;
        cv::Point maxLoc;
        cv::minMaxLoc(response_cf_part, nullptr, &maxVal, nullptr, &maxLoc);

        // valid part
        double normed_max_val =  1.0 / (1.0 + std::exp(-maxVal));
        valid_part[p] = (normed_max_val > cfg.part_detect_valid_thresh);

        float centerx = (1 + norm_delta_area_part.width) / 2 - 1;
        float centery = (1 + norm_delta_area_part.height) / 2 - 1;

        rect_part[p].x += (maxLoc.x - centerx) / area_resize_factor_part;
        rect_part[p].y += (maxLoc.y - centery) / area_resize_factor_part;
    }
    // scale estimate based on part distance
    cv::Point_<float> center_part1(rect_part[0].x + rect_part[0].width / 2.0, rect_part[0].y + rect_part[0].height / 2.0);
    cv::Point_<float> center_part2(rect_part[1].x + rect_part[1].width / 2.0, rect_part[1].y + rect_part[1].height / 2.0);
    double curr_part_dist = sqrt(pow(center_part1.x - center_part2.x, 2) + pow(center_part1.y - center_part2.y, 2));
    if(valid_part[0] && valid_part[1])
        curr_scale_based_on_part = curr_part_dist / init_part_dist;

    if(is_tall){
        part_rect1.width = part_rect2.width = round(base_target_sz.width * curr_scale_based_on_part);
        part_rect1.height = part_rect2.height = round(0.5 * base_target_sz.height * curr_scale_based_on_part);
    }
    else{
        part_rect1.width = part_rect2.width = round(0.5 * base_target_sz.width * curr_scale_based_on_part);
        part_rect1.height = part_rect2.height = round(base_target_sz.height * curr_scale_based_on_part);
    }
    part_rect1.x = center_part1.x - 0.5 * part_rect1.width;
    part_rect1.y = center_part1.y - 0.5 * part_rect1.height;
    part_rect2.x = center_part2.x - 0.5 * part_rect2.width;
    part_rect2.y = center_part2.y - 0.5 * part_rect2.height;

    // global center rough estimate
    if(valid_part[0] && valid_part[1]){
        pos.x = 0.5 * (center_part1.x + center_part2.x);
        pos.y = 0.5 * (center_part1.y + center_part2.y);
    }
}


// TESTING step
cv::Rect STAPLE_TRACKER::tracker_staple_update(cv::Mat im, cv::Rect_<float> &part_rect1, cv::Rect_<float> &part_rect2){
    if(cfg.use_part_track)
        part_update(im, part_rect1, part_rect2);

    // global center detect 
    // extract patch of size bg_area and resize to norm_bg_area
    cv::Mat im_patch_cf;
    getSubwindow(im, pos, norm_bg_area, bg_area, im_patch_cf);

    // compute feature map. fhog feature
    cv::MatND xt_windowed;
    getFeatureMap(im_patch_cf, cfg.feature_type, xt_windowed);

    // compute FFT
    std::vector<cv::Mat> xtsplit;
    matsplit(xt_windowed, xtsplit);
    
    std::vector<cv::Mat> xtf;
    for (int i =  0; i < xt_windowed.channels(); i++) {
        cv::Mat dimf;
        cv::dft(xtsplit[i], dimf);
        xtf.push_back(dimf);
    }

    std::vector<cv::Mat> hf;
    const int w = xt_windowed.cols;
    const int h = xt_windowed.rows;

    // Correlation between filter and test patch gives the response
    // Solve diagonal system per pixel.
    // F*G/F*F
    if (cfg.den_per_channel){
        for (int ch = 0; ch < xt_windowed.channels(); ++ch){
            cv::Mat dim = cv::Mat(h, w, CV_32FC2);
            for (int j = 0; j < h; ++j){
                const cv::Vec2f* pSrc = hf_num[ch].ptr<cv::Vec2f>(j);
                const float* pDen = hf_den[ch].ptr<float>(j);
                cv::Vec2f* pDst = dim.ptr<cv::Vec2f>(j);
                for (int i = 0; i < w; ++i)
                    pDst[i] = pSrc[i] / (pDen[i] + cfg.lambda);
            }
            hf.push_back(dim);
        }
    }
    else{
        // merge each channel of hf_den to one channel
        std::vector<float> DIM1(w * h, cfg.lambda);

        for (int ch = 0; ch < xt_windowed.channels(); ++ch){
            float* pDim1 = &DIM1[0];
            for (int j = 0; j < h; ++j){
                const float* pDen = hf_den[ch].ptr<float>(j);
                for (int i = 0; i < w; ++i){
                    *pDim1 += pDen[i];
                    ++pDim1;
                }
            }
        }

        for (int ch = 0; ch < xt_windowed.channels(); ++ch){
            cv::Mat dim = cv::Mat(h, w, CV_32FC2);
            const float* pDim1 = &DIM1[0];
            for (int j = 0; j < h; ++j){
                const cv::Vec2f* pSrc = hf_num[ch].ptr<cv::Vec2f>(j);
                cv::Vec2f* pDst = dim.ptr<cv::Vec2f>(j);
                for (int i = 0; i < w; ++i){
                    *pDst = *pSrc / *pDim1;
                    ++pDim1;
                    ++pDst;
                    ++pSrc;
                }
            }
            hf.push_back(dim);
        }
    }

    // Z*(F*G/F*F)
    cv::Mat response_cff = cv::Mat(h, w, CV_32FC2);
    for (int j = 0; j < h; j++){
        cv::Vec2f* pDst = response_cff.ptr<cv::Vec2f>(j);
        for (int i = 0; i < w; i++){
            float sum = 0.0;
            float sumi = 0.0;
            for (size_t ch = 0; ch < hf.size(); ch++){
                cv::Vec2f pHF = hf[ch].at<cv::Vec2f>(j,i);
                cv::Vec2f pXTF = xtf[ch].at<cv::Vec2f>(j,i);

                sum += (pHF[0] * pXTF[0] + pHF[1] * pXTF[1]);
                sumi += (pHF[0] * pXTF[1] - pHF[1] * pXTF[0]);
            }
            *pDst = cv::Vec2f(sum, sumi);
            ++pDst;
        }
    }

    cv::Mat response_cfi;
    cv::dft(response_cff, response_cfi, cv::DFT_SCALE|cv::DFT_INVERSE);
    cv::Mat response_cf = ensure_real(response_cfi);

    // Crop square search region (in feature pixels).
    cv::Size newsz = norm_delta_area;
    newsz.width = floor(newsz.width / cfg.hog_cell_size);
    newsz.height = floor(newsz.height / cfg.hog_cell_size);

    (newsz.width % 2 == 0) && (newsz.width -= 1);
    (newsz.height % 2 == 0) && (newsz.height -= 1);

    cropFilterResponse(response_cf, newsz, response_cf);

    if (cfg.hog_cell_size > 1){
        cv::Mat temp;
        mexResize(response_cf, temp, norm_delta_area, "auto");
        response_cf = temp;
    }
    // POSITION ESTIMATION
    cv::Mat response = response_cf;
    
    if(cfg.use_color_hist_model){
        cv::Size pwp_search_area;
        pwp_search_area.width = round(norm_pwp_search_area.width / area_resize_factor);
        pwp_search_area.height = round(norm_pwp_search_area.height / area_resize_factor);
        
        cv::Mat im_patch_pwp;

        getSubwindow(im, pos, norm_pwp_search_area, pwp_search_area, im_patch_pwp);
        
        cv::Mat likelihood_map;
        // get (h, w, 1) as im_patch_pwp size
        getColourMap(im_patch_pwp, likelihood_map);

        // each pixel of response_pwp loosely represents the likelihood that
        // the target (of size norm_target_sz) is centred on it
        cv::Mat response_pwp;
        getCenterLikelihood(likelihood_map, norm_target_sz, response_pwp);
        mergeResponses(response_cf, response_pwp, response);
    }

    // APCE judgement
    if(cfg.use_apce_jugment){
        double curr_apce_val = 0, curr_F_max = 0;
        calAPCE(response, curr_apce_val, curr_F_max);
        num_apce += 1;

        mean_apce = (double)(num_apce - 1) / num_apce * mean_apce + curr_apce_val / num_apce;
        mean_fmax = (double)(num_apce - 1) / num_apce * mean_fmax + curr_F_max / num_apce;
        apce_update = (curr_apce_val >= cfg.beta1 * mean_apce) && (curr_F_max >= cfg.beta2 * mean_fmax);
        // std::cout << "curr apce:" << curr_apce_val << " ";
        // std::cout << "mean apce:" << mean_apce << " ";
        // std::cout << "curr Fmax:" << curr_F_max << " ";
        // std::cout << "mean Fmax:" << mean_fmax << " ";
        // std::cout << "apce update:" << apce_update << std::endl;
    }
    
    double maxVal = 0;
    cv::Point maxLoc;

    cv::minMaxLoc(response, nullptr, &maxVal, nullptr, &maxLoc);

    float centerx = (1 + norm_delta_area.width) / 2 - 1;
    float centery = (1 + norm_delta_area.height) / 2 - 1;

    pos.x += (maxLoc.x - centerx) / area_resize_factor;
    pos.y += (maxLoc.y - centery) / area_resize_factor;

    cv::Rect_<float> location;

    location.x = pos.x - target_sz.width/2.0;
    location.y = pos.y - target_sz.height/2.0;
    location.width = target_sz.width;
    location.height = target_sz.height;

    // SCALE SPACE SEARCH
    if (cfg.scale_adaptation){
        cv::Mat im_patch_scale;
        getScaleSubwindow(im, pos, im_patch_scale);

        cv::Mat xsf;
        cv::dft(im_patch_scale, xsf, cv::DFT_ROWS);

        const int w = xsf.cols;
        const int h = xsf.rows;

        cv::Mat scale_responsef = cv::Mat(1, w, CV_32FC2, cv::Scalar(0, 0, 0));
        for (int j = 0; j < h; ++j){
            const float* pXSF = xsf.ptr<float>(j);
            const float* pXSFNUM = sf_num.ptr<float>(j);
            const float* pDen = sf_den.ptr<float>(0);
            float* pscale = scale_responsef.ptr<float>(0);

            for (int i = 0; i < w; ++i){
                float invDen = 1.f / (*pDen + cfg.lambda);

                pscale[0] += invDen * (pXSFNUM[0]*pXSF[0] - pXSFNUM[1]*pXSF[1]);
                pscale[1] += invDen * (pXSFNUM[0]*pXSF[1] + pXSFNUM[1]*pXSF[0]);

                pscale += 2;
                pXSF += 2;
                pXSFNUM += 2;
                ++pDen;
            }
        }

        cv::Mat scale_response;
        cv::dft(scale_responsef, scale_response, cv::DFT_SCALE|cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);

        double maxVal = 0;
        cv::Point maxLoc;
        cv::minMaxLoc(scale_response, nullptr, &maxVal, nullptr, &maxLoc);

        int recovered_scale =  maxLoc.x;

        // std::cout << "cur scale factor " << scale_factors.at<float>(recovered_scale) << std::endl;
        scale_factor = scale_factor * scale_factors.at<float>(recovered_scale);
        // scale_factor = curr_scale_based_on_part;

        // merge cf scale estimate and part based scale estimate results
        if(cfg.use_part_track && valid_part[0] && valid_part[1])
            scale_factor = scale_factor * (1 - cfg.merge_factor_scale) + curr_scale_based_on_part * cfg.merge_factor_scale;

        // if (scale_factor < min_scale_factor)
        //     scale_factor = min_scale_factor;
        // else if (scale_factor > max_scale_factor)
        //     scale_factor = max_scale_factor;

        // use new scale to update bboxes for target, filter, bg and fg models
        target_sz.width = round(base_target_sz.width * scale_factor);
        target_sz.height = round(base_target_sz.height * scale_factor);

        location.x = pos.x - target_sz.width/2.0;
        location.y = pos.y - target_sz.height/2.0;
        location.width = target_sz.width;
        location.height = target_sz.height;

        double avg_dim = (target_sz.width + target_sz.height)/2.0;

        bg_area.width= round(target_sz.width + avg_dim);
        bg_area.height = round(target_sz.height + avg_dim);

        if(bg_area.width > im.cols)
            bg_area.width = im.cols - 1;
        if(bg_area.height > im.rows)
            bg_area.height = im.rows - 1;

        bg_area.width = bg_area.width - (bg_area.width - target_sz.width) % 2;
        bg_area.height = bg_area.height - (bg_area.height - target_sz.height) % 2;

        fg_area.width = round(target_sz.width - avg_dim * cfg.inner_padding);
        fg_area.height = round(target_sz.height - avg_dim * cfg.inner_padding);

        fg_area.width = fg_area.width + int(bg_area.width - fg_area.width) % 2;
        fg_area.height = fg_area.height + int(bg_area.height - fg_area.height) % 2;

        // Compute the rectangle with (or close to) params.fixed_area and
        // same aspect ratio as the target bboxgetScaleSubwindow
        area_resize_factor = sqrt(cfg.fixed_area / (float)(bg_area.width * bg_area.height));

        if(cfg.use_part_track){
            // update bboxes for target, filter bg_area for part
            if(is_tall){
                cv::Rect_<float> rect_part1(location.x, location.y, location.width, location.height / 2.0);
                cv::Rect_<float> rect_part2(location.x, location.y + location.height / 2.0, location.width, location.height / 2.0);
                rect_part[0] = rect_part1;
                rect_part[1] = rect_part2;
            }
            else{
                cv::Rect_<float> rect_part1(location.x, location.y, location.width / 2.0, location.height);
                cv::Rect_<float> rect_part2(location.x + location.width / 2.0, location.y, location.width / 2.0, location.height);
                rect_part[0] = rect_part1;
                rect_part[1] = rect_part2;
            }
            cv::Size part_sz(rect_part[0].width, rect_part[0].height);
            double avg_dim_part = (part_sz.width + part_sz.height) / 2.0;

            bg_area_part.width = round(part_sz.width + avg_dim_part);
            bg_area_part.height = round(part_sz.height + avg_dim_part);

            bg_area_part.width = std::min(bg_area_part.width, im.size().width - 1);
            bg_area_part.height = std::min(bg_area_part.height, im.size().height - 1);

            bg_area_part.width = bg_area_part.width - (bg_area_part.width - part_sz.width) % 2;
            bg_area_part.height = bg_area_part.height - (bg_area_part.height - part_sz.height) % 2;

            area_resize_factor_part = sqrt(cfg.fixed_area_part / double(bg_area_part.width * bg_area_part.height));
        }
    }

    return location;
}