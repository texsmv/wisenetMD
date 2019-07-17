#ifndef WISENETMD_H
#define WISENETMD_H

#include "includes.h"
#include "videoInput.h"
#include "kernels.h"





/**
 * @brief 
 * 
 * R_texture = R_lbsp
 */
class WisenetMD
{
private:    
    uchar* h_B_color; int* h_B_texture; /* host background samples, dimension->(height, width, n_B)*/
    uchar* d_B_color; int* d_B_texture; /* device background samples, dimension->(height, width, n_B) */
    uchar* h_DB; /* host dynamic background samples, dimension->(height, width, n_DB) */
    uchar* d_DB; /* device dynamic background samples, dimension->(height, width, n_DB) */
    float* h_T; float* d_T; /* host/device update paramaters, dimension->(height, width)*/
    float* h_R_color; float* h_R_texture; /*host distance thresholds, dimension->(height, width)*/
    float* d_R_color; float* d_R_texture; /*device distance thresholds, dimension->(height, width)*/
    bool* h_S;  bool* d_S; /* host/device S mask*/
    bool* h_S_l;  bool* d_S_l; /* host/device S last mask*/
    bool* h_DR; bool* d_DR; /* host/device mask of Dynamic Region */
    float* h_BR; float* d_BR; /* host/device Blinking Rate*/
    float* h_TB; float* d_TB; /* host/ device Total number of Blinking pixels*/
    float* h_Dist_last; float* d_Dist_last; /* host/ device distance to last frame*/
    float* h_S_feed; float* d_S_feed; /*host/ device trajectory of object*/
    float* h_d; float* d_d; /*host/ device min distance of pixel to its samples*/
    float* h_D_min; float* d_D_min; /*todo: complete,   host/ device  ..*/
    float* h_v; float* d_v;
    float* h_w; float* d_w;
    float* h_R; float* d_R;

    float ratio_movement;
    

    uchar* d_color_l;
    int* d_texture_l;

    cv::Mat m_S, m_DR, m_T, m_R, m_BR, m_v, m_D_min, m_d, m_Dist_last, m_S_feed, m_w, m_R_color, m_R_texture, m_TB;


    // random numbers
    curandState_t* states;
    float alpha = 0.04;
    float Ro_color = 50; float Ro_texture = 3; /* initial thresholds */
    float blink_threshold = 0.05;   
    float v_decr = 0.1;
    unsigned int n_B = 50;
    unsigned int n_DB = 30;
    unsigned int t = 0;
    float DB_color_threshold = 30;
    VideoInput* input;

    dim3 block, grid;
    unsigned int width, height;
public:
    WisenetMD();
    ~WisenetMD();


    
    void init();
    void init_model();

    /* calculate mask, update model */
    void process_frame();

    /* calculate current d_S */
    void calc_S();

    /* update background samples*/
    void update_B();

    void save_last_values();

    /* getters and setters */
    void set_input(VideoInput* input);

    float get_movement_ratio();
};





#endif
