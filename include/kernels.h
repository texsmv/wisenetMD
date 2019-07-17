#ifndef KERNELS2_H
#define KERNELS2_H

#include "includes.h"

/*------------------------Kernels for wisenetMD ----------------------*/

namespace kernels{
    /* initialize background samples (suppose 2d block and grid)*/
    __global__ void init_B_samples(uchar* d_B_color, int* d_B_texture, unsigned int n_B, uchar* d_rgb, int* d_texture, unsigned int height, unsigned int width);

    __global__ void init_DB_samples(uchar* d_DB, unsigned int n_DB, uchar* d_rgb, unsigned int height, unsigned int width);
    
    __global__ void calc_R_color_texture(float* d_R, float* d_R_color, float* d_R_texture, float Ro_color, float Ro_texture, unsigned int height, unsigned int width);
    
    __global__ void calc_S(bool* d_S, uchar* d_B_color, int* d_B_texture, unsigned int n_B, float* d_R_color, float* d_R_texture, uchar* d_rgb, int* d_texture, unsigned int height, unsigned int width);

    __global__ void update_B(bool* d_S, uchar* d_B_color, int* d_B_texture, unsigned int n_B, uchar* d_rgb, int* d_texture, float* d_T, unsigned int height, unsigned int width, curandState_t* states);

    __global__ void init_randoms(unsigned int seed, curandState_t* states, int h, int w);

    __global__ void calc_DR(bool* d_S, bool* d_S_l, bool* d_DR, float* d_BR, float* d_TB, unsigned int t, float blink_threshold, unsigned int height, unsigned int width);

    __global__ void calc_Dist_last(float* d_Dist_last, uchar* d_rgb, int* d_texture, uchar* d_rgb_l, int* d_texture_l, unsigned int height, unsigned int width);

    __global__ void calc_S_feed(float* d_S_feed, bool* d_S_l, float alpha, unsigned int height, unsigned int width);

    __global__ void update_DB(uchar* d_DB, unsigned int n_DB, bool* d_DR, bool* d_S, bool* d_S_l, float* d_Dist_last, float* d_S_feed, uchar* d_rgb, unsigned int height, unsigned int width, curandState_t* states);

    __global__ void calc_d(float* d_d, uchar* d_B_color, int* d_B_texture, unsigned int n_B, float* d_R_color, float* d_R_texture, uchar* d_rgb, int* d_texture, unsigned int height, unsigned int width);

    __global__ void calc_D_min(float* d_D_min, float* d_d, float alpha, unsigned int height, unsigned int width);
    
    __global__ void calc_w(float* d_w, bool* d_DR, float* d_Dist_last, float* d_S_feed, unsigned int height, unsigned int width);

    __global__ void calc_v(float* d_v, float v_decr, bool* d_S, bool* d_S_l, float* d_w, unsigned int height, unsigned int width);

    __global__ void update_R(float* d_R, float* d_v, float* d_D_min, unsigned int height, unsigned int width);

    __global__ void update_T(float* d_T, bool* d_S, float* d_v, float* d_D_min, unsigned int height, unsigned int width);

    __global__ void FP_recheck(uchar* d_DB, unsigned int n_DB, bool* d_DR, bool* d_S, uchar* d_rgb, float DB_color_threshold, unsigned int height, unsigned int width);

}   



/*--------------------------device functions---------------------------*/
__device__ int HammingDist(int x, int y);
__device__ int l1Dist(uchar* B_color, uchar* I_color);
__device__ int random(curandState_t* states, int h, int w, int i, int j,int min, int max);
__device__ float clip(float i, float a, float b);


#endif