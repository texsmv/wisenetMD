#ifndef TEXTURES_H
#define TEXTURES_H

#include "includes.h"
__device__ int lbsp_pixel(uchar* mat, int h, int w, int j, int i, float tau);
__device__ int olbp_pixel(uchar* mat, int h, int w, int i, int j, float tau);
__device__ int lsbp_pixel(uchar* mat, int h, int w, int j, int i, float tau);
__device__ int xcslbp_pixel(uchar* mat, int h, int w, int xc, int yc, float tau);
__global__ void texture_kernel(uchar* d_mat, int* d_lbp, float tau, int h, int w, int texture_number);
void cuda_texture(uchar* d_mat, int* d_texture, int h, int w, float tau, dim3 block, dim3 grid, int texture_number);
#endif