#include "videoInput.h"

VideoInput::VideoInput(/* args */)
{
}

VideoInput::~VideoInput()
{
}


VideoInput::VideoInput(std::string path_imgs)
{
    /* initializing variables*/
    cap = new cv::VideoCapture(path_imgs);
    // cap = new cv::VideoCapture(0);
    width = cap->get(cv::CAP_PROP_FRAME_WIDTH);
    height = cap->get(cv::CAP_PROP_FRAME_HEIGHT);
    count_frames = cap->get(cv::CAP_PROP_FRAME_COUNT);
    // count_frames = 1000;
    size.height = height;    size.width = width;
    
    rgb_data_size = width * height * 3;
    gray_data_size = width * height;
    
    /* initializing arrays */
    d_gray = cudu::array<uchar>(gray_data_size);
    d_rgb = cudu::array<uchar>(rgb_data_size);
    h_texture = new int[gray_data_size];
    d_texture = cudu::array<int>(gray_data_size);
    


    /* initializing cv::mats */
    texture_mat = cv::Mat(height, width, CV_8UC3);

    block = dim3(16, 16, 1);
    grid = dim3(ceil( height/ float(block.x)), ceil(width / float(block.y)));
}


void VideoInput::resize(cv::Size n_size){
    b_resize = true;
    width = n_size.width;
    height = n_size.height;
    this->size.height = height;    this->size.width = width;

    /*todo: change this*/
    rgb_data_size = width * height * 3;
    gray_data_size = width * height;
    
    /* initializing arrays */
    d_gray = cudu::array<uchar>(gray_data_size);
    d_rgb = cudu::array<uchar>(rgb_data_size);
    h_texture = new int[gray_data_size];
    d_texture = cudu::array<int>(gray_data_size);
    


    /* initializing cv::mats */
    texture_mat = cv::Mat(height, width, CV_8UC3);

    block = dim3(16, 16, 1);
    grid = dim3(ceil( height/ float(block.x)), ceil(width / float(block.y)));
}






void VideoInput::grab_frame(){
    if(curr_count < count_frames){
        (*cap) >> curr_frame;
        /*resize if needed */
        if(b_resize)    cv::resize(curr_frame, curr_frame, size);
        /*convert to gray */
        curr_gray_frame = cv::Mat(height, width, CV_8UC1);
        cv::cvtColor(curr_frame, curr_gray_frame, CV_BGR2GRAY);
        
        /*get host pointer to gray data */
        h_gray = (uchar*)(curr_gray_frame.data);
        /*copy and get pointer of gray data in device */
        cudu::h2d<uchar>(h_gray, d_gray ,gray_data_size);
        
        /*get host pointer to rgb data */
        h_rgb = (uchar*)(curr_frame.data);
        /*copy and get pointer of rgb data in device */
        cudu::h2d<uchar>(h_rgb, d_rgb, rgb_data_size);
        
        /*calculate texture from gray data*/
        /*todo: change tau*/
        float tau = 15;
        cuda_texture(d_gray, d_texture, height, width, tau, block, grid, 3);
        devs();
        cudu::d2h<int>(d_texture, h_texture, gray_data_size);
        devs();
        // pow(2, 16) - 1
        cvu::T2mat<int>(texture_mat, height, width, h_texture, pow(2, 16) - 1);
        

        curr_count++;
    }
}




bool VideoInput::is_empty(){
    if(curr_count < count_frames)
        return false;
    return true;
}

void VideoInput::get_mat(cv::Mat& mat){
    mat = curr_frame;
}

void VideoInput::get_texture_mat(cv::Mat& mat){
    mat = texture_mat;
}

cv::Size VideoInput::get_size(){
    return size;
}

unsigned int VideoInput::get_count(){
    return count_frames;
}


