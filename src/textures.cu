#include "textures.h"


__device__ int lbsp16_pixel(uchar* mat, int h, int w, int j, int i){
    float center_pixel = at2(mat, j, i, w);
    int sum = 0;
    float tr = 0.2;
  
    int neighbors[16];
    neighbors[0] = at2(mat, j - 2, i - 2, w);
    neighbors[1] = at2(mat, j - 2, i, w);
    neighbors[2] = at2(mat, j - 2, i + 2, w);
    neighbors[3] = at2(mat, j , i + 2, w);
    neighbors[4] = at2(mat, j + 2, i + 2, w);
    neighbors[5] = at2(mat, j + 2, i, w);
    neighbors[6] = at2(mat, j + 2, i - 2, w);
    neighbors[7] = at2(mat, j , i - 2, w);

    neighbors[8] = at2(mat, j - 1, i - 1, w);
    neighbors[9] = at2(mat, j - 1, i , w);
    neighbors[10] = at2(mat, j - 1, i + 1, w);
    neighbors[11] = at2(mat, j , i + 1, w);
    neighbors[12] = at2(mat, j + 1, i + 1, w);
    neighbors[13] = at2(mat, j + 1, i , w);
    neighbors[14] = at2(mat, j + 1, i - 1, w);
    neighbors[15] = at2(mat, j , i - 1, w);
    

  
    for(int k = 0; k < 16; k++){
      if (fabs(neighbors[k]-center_pixel) < (tr * center_pixel))
        sum += pow(2, k);
    }
  
    return sum;
  }
  


__device__ int lsbp_pixel(uchar* mat, int h, int w, int j, int i, float tau){
    float center_pixel = at2(mat, j, i, w);
    int sum = 0;
  
  
    int vXYNeighbor[8];
    int fxRadius = 1;
    int fyRadius = 1;
    int xyNeighborPoints = 8;
    // store the neighbors in a vector
    for (int k = 0; k < 8; k++)
    {
      float x = floor((double)j + (double)fxRadius * cos((2 * M_PI * k) / xyNeighborPoints) + 0.5);
      float y = floor((double)i - (double)fyRadius * sin((2 * M_PI * k) / xyNeighborPoints) + 0.5);
  
      vXYNeighbor[k] = at2(mat, (int)x, (int)y, w);
    }
  
    for(int k = 0; k < xyNeighborPoints; k++){
      if (fabs(vXYNeighbor[k]-center_pixel) < tau)
        sum += pow(2, k);
    }
  
    return sum;
  }
  
  __device__ int olbp_pixel(uchar* mat, int h, int w, int i, int j, float tau){
    float center_pixel = at2(mat, i, j, w);
    int sum = 0;
  
  
    int vXYNeighbor[8];
    int fxRadius = 1;
    int fyRadius = 1;
    int xyNeighborPoints = 8;
    // store the neighbors in a vector
    for (int k = 0; k < 8; k++)
    {
      float x = floor((double)i + (double)fxRadius * cos((2 * M_PI * k) / xyNeighborPoints) + 0.5);
      float y = floor((double)j - (double)fyRadius * sin((2 * M_PI * k) / xyNeighborPoints) + 0.5);
  
      vXYNeighbor[k] = at2(mat, (int)x, (int)y, w);
    }
  
    for(int k = 0; k < xyNeighborPoints; k++){
      if (vXYNeighbor[k] > center_pixel)
        sum += pow(2, k);
    }
  
    return sum;
  }
  
  __device__ int xcslbp_pixel(uchar* mat, int h, int w, int xc, int yc, float tau){
    int xyNeighborPoints = 8;
    int fxRadius = 1;
    int fyRadius = 1;
  
    float p1XY, pXY, pcXY;
    // int centerVal = gray.data[xc + yc * gray.cols];
    float centerVal = at2(mat, xc, yc, w);
  
    // XY plane
    int basicLBP = 0;
    int featureBin = 0;
    int countXY = 0;
    int vXYNeighbor[8];
  
    // store the neighbors in a vector
    for (int k = 0; k < xyNeighborPoints; k++)
    {
      float x = floor((double)xc + (double)fxRadius * cos((2 * M_PI * k) / xyNeighborPoints) + 0.5);
      float y = floor((double)yc - (double)fyRadius * sin((2 * M_PI * k) / xyNeighborPoints) + 0.5);
  
      // vXYNeighbor[k] = gray.data[(int)x + (int)y * gray.cols];
      vXYNeighbor[k] = at2(mat, (int)x, (int)y, w);
      countXY++;
    }
  
    // loop to calculate XCSLBP
    for (int kXY = 0; kXY < 4; kXY++)
    {
      if (kXY == 0)
      {
        p1XY = vXYNeighbor[0] - vXYNeighbor[4];
        pXY = vXYNeighbor[0] - centerVal;
        pcXY = vXYNeighbor[4] - centerVal;
      }
  
      if (kXY == 1)
      {
        p1XY = vXYNeighbor[7] - vXYNeighbor[3];
        pXY = vXYNeighbor[2] - centerVal;
        pcXY = vXYNeighbor[6] - centerVal;
      }
  
      if (kXY == 2)
      {
        p1XY = vXYNeighbor[6] - vXYNeighbor[2];
        pXY = vXYNeighbor[1] - centerVal;
        pcXY = vXYNeighbor[5] - centerVal;
      }
  
      if (kXY == 3)
      {
        p1XY = vXYNeighbor[5] - vXYNeighbor[1];
        pXY = vXYNeighbor[3] - centerVal;
        pcXY = vXYNeighbor[7] - centerVal;
      }
  
      float currentVal = (p1XY + centerVal) + (pXY*pcXY);
  
      if (currentVal <= 0)
        basicLBP += pow(2.0, featureBin);
  
      featureBin++;
    }
    // save pixel in output
    return 256 - basicLBP;
    // return basicLBP;
    // histogram.data[basicLBP] += 1;
  
  }
  

// lsbp kernel
__global__ void texture_kernel(uchar* d_mat, int* d_lbp, float tau, int h, int w, int texture_number){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row > 0 && row < (h - 1) && col < (w - 1) && col > 0){
        if(texture_number == 0)
            at2(d_lbp, row, col, w) = olbp_pixel(d_mat, h, w, row, col, tau);
        else if(texture_number == 1)
            at2(d_lbp, row, col, w) = lsbp_pixel(d_mat, h, w, row, col, tau);
        else if(texture_number == 2)
            at2(d_lbp, row, col, w) = xcslbp_pixel(d_mat, h, w, row, col, tau);
        else if(texture_number == 3)
            at2(d_lbp, row, col, w) = lbsp16_pixel(d_mat, h, w, row, col);
    }
  
  }
  
  
  // function to call kernel
void cuda_texture(uchar* d_mat, int* d_texture, int h, int w, float tau, dim3 block, dim3 grid, int texture_number){
    texture_kernel<<<grid, block>>>(d_mat, d_texture, tau, h, w, texture_number);
}
  