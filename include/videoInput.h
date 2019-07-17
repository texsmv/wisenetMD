#ifndef VIDEO_INPUT_H
#define VIDEO_INPUT_H

#include "includes.h"
#include "textures.h"

/**
 * @brief gets all input data from a video 
 * 
 * the way to get pixel value is j, i(height, width)
 * 
 */


class VideoInput
{
private:
    cv::VideoCapture * cap;
    cv::Size size;
    cv::Mat curr_frame; /*rgb mat of current frame*/    
    cv::Mat curr_gray_frame; /*gray mat of current frame*/    
    uchar* h_rgb; /*pointer to host rgb data of current frame*/
    uchar* h_gray; /*pointer to host gray data of current frame*/
    uchar* d_rgb; /*pointer to device rgb data of current frame*/
    uchar* d_gray; /*pointer to device gray data of current frame*/
    int* h_texture; /*pointer to host texture data of current frame*/
    int* d_texture; /*pointer to device texture data of current frame*/

    unsigned int rgb_data_size;
    unsigned int gray_data_size;


    unsigned int width, height;
    unsigned int count_frames, curr_count = 0;
    bool b_resize = false;
    dim3 block, grid;

    cv::Mat texture_mat;
    
    

    friend class WisenetMD;

    
public:
    VideoInput();
    VideoInput(std::string path_imgs);
    ~VideoInput();

    /*set and gets */
    cv::Size get_size();
    unsigned int get_count();
    /**
     * @brief Get the mat object by reference
     * 
     * @param mat 
     */
    void get_mat(cv::Mat& mat);

    /**
     * @brief Get the texture mat object by reference
     * 
     * @param mat 
     */
     void get_texture_mat(cv::Mat& mat);






    
    /* optional - set new size */
    void resize(cv::Size size);


    /* process new frame from cap */
    void grab_frame();


    /**
     * @brief is cap empty 
     * - not for cameras
     * 
     * @return true if is empty
     * @return false otherwise
     */
    bool is_empty();


    /**
     * @brief calculate texture of current frame 
     * - d_curr_data_gray is needed
     * 
     * 
     */
    void calc_texture();






    





    
    
};







#endif
