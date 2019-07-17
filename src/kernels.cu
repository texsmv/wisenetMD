#include "kernels.h"

/*------------------------Kernels for wisenetMD ----------------------*/


__global__ void kernels::init_B_samples(uchar* d_B_color, int* d_B_texture, unsigned int n_B, uchar* d_rgb, int* d_texture, unsigned int height, unsigned int width){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(j < height && i < width){
        for(int k = 0; k < n_B; k++){
            at4(d_B_color, j, i, k, 0, width, n_B, 3) = at3(d_rgb, j, i, 0, width, 3);
            at4(d_B_color, j, i, k, 1, width, n_B, 3) = at3(d_rgb, j, i, 1, width, 3);
            at4(d_B_color, j, i, k, 2, width, n_B, 3) = at3(d_rgb, j, i, 2, width, 3);

            at3(d_B_texture, j, i, k, width, n_B) = at2(d_texture, j, i, width);
        }
    }
}

__global__ void kernels::init_DB_samples(uchar* d_DB, unsigned int n_DB, uchar* d_rgb, unsigned int height, unsigned int width){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(j < height && i < width){
        for(int k = 0; k < n_DB; k++){
            at4(d_DB, j, i, k, 0, width, n_DB, 3) = at3(d_rgb, j, i, 0, width, 3);
            at4(d_DB, j, i, k, 1, width, n_DB, 3) = at3(d_rgb, j, i, 1, width, 3);
            at4(d_DB, j, i, k, 2, width, n_DB, 3) = at3(d_rgb, j, i, 2, width, 3);
        }
    }
}

__global__ void kernels::calc_S(bool* d_S, uchar* d_B_color, int* d_B_texture, unsigned int n_B, float* d_R_color, float* d_R_texture, uchar* d_rgb, int* d_texture, unsigned int height, unsigned int width){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    bool b_l1Dist;
    bool b_hammingDist;
    if(j < height && i < width){
        int N = 0;
        for(int k = 0; k < n_B; k++){
            b_l1Dist =  (l1Dist( &(at4(d_B_color, j, i, k, 0, width, n_B, 3)), &(at3(d_rgb, j, i, 0, width, 3))) < at2(d_R_color, j, i, width));
            b_hammingDist =  (HammingDist( at3(d_B_texture, j, i, k, width, n_B), at2(d_texture, j, i, width)) <  at2(d_R_texture, j, i, width));
            if(b_l1Dist && b_hammingDist)
                N++;
        }
        if(N < 2)
            at2(d_S, j, i, width) = true;
        else
            at2(d_S, j, i, width) = false;
    }
}



__global__ void kernels::update_B(bool* d_S, uchar* d_B_color, int* d_B_texture, unsigned int n_B, uchar* d_rgb, int* d_texture, float* d_T, unsigned int height, unsigned int width,  curandState_t* states){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(j < height && i < width) if ( at2(d_S, j, i, width) == false){
        if((random(states, j, i, height, width, 0, 100) / 100.0) < (1.0 / at2(d_T, j, i, width))){
            int r = random(states, j, i, height, width, 0, n_B);
            at4(d_B_color, j, i, r, 0, width, n_B, 3) = at3(d_rgb, j, i, 0, width, 3);
            at4(d_B_color, j, i, r, 1, width, n_B, 3) = at3(d_rgb, j, i, 1, width, 3);
            at4(d_B_color, j, i, r, 2, width, n_B, 3) = at3(d_rgb, j, i, 2, width, 3);

            at3(d_B_texture, j, i, r, width, n_B) = at2(d_texture, j, i, width);
        }
    }


}




__global__ void kernels::calc_DR(bool* d_S, bool* d_S_l, bool* d_DR, float* d_BR, float* d_TB, unsigned int t, float blink_threshold, unsigned int height, unsigned int width){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(j < height && i < width){
        if ( at2(d_S, j, i, width) != at2(d_S_l, j, i, width)){
            at2(d_TB, j, i, width) = (at2(d_TB, j, i, width) + 1);
        }
        at2(d_BR, j, i, width) = (at2(d_TB, j, i, width) / float(t));
        if(at2(d_BR, j, i, width) > blink_threshold)
            at2(d_DR, j, i, width) = true;
        else
            at2(d_DR, j, i, width) = false;

    }
}


__global__ void kernels::calc_Dist_last(float* d_Dist_last, uchar* d_rgb, int* d_texture, uchar* d_rgb_l, int* d_texture_l, unsigned int height, unsigned int width){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(j < height && i < width){
        at2(d_Dist_last, j, i, width) = ((l1Dist( &(at3(d_rgb, j, i, 0, width, 3)), &(at3(d_rgb_l, j, i, 0, width, 3))) / (255.0 * 3) +
        HammingDist( at2(d_texture, j, i, width), at2(d_texture_l, j, i, width)) / (16.0 * 3)) / 2.0);
    }
}


__global__ void kernels::calc_S_feed(float* d_S_feed, bool* d_S_l, float alpha, unsigned int height, unsigned int width){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(j < height && i < width){
        at2(d_S_feed, j, i, width) = (1 - alpha) * at2(d_S_feed, j, i, width) + (alpha / 255.0) * float(at2(d_S_l, j, i, width));
    }
}

__global__ void kernels::update_DB(uchar* d_DB, unsigned int n_DB, bool* d_DR, bool* d_S, bool* d_S_l, float* d_Dist_last, float* d_S_feed, uchar* d_rgb, unsigned int height, unsigned int width, curandState_t* states){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(j < height && i < width){
        if ( at2(d_S, j, i, width) != at2(d_S_l, j, i, width)){
            if((at2(d_Dist_last, j, i, width) > 0.45) &&  (at2(d_S_feed, j, i, width) < 0.4)){
                int r = random(states, j, i, height, width, 0, n_DB);
                at4(d_DB, j, i, r, 0, width, n_DB, 3) = at3(d_rgb, j, i, 0, width, 3);
                at4(d_DB, j, i, r, 1, width, n_DB, 3) = at3(d_rgb, j, i, 1, width, 3);
                at4(d_DB, j, i, r, 2, width, n_DB, 3) = at3(d_rgb, j, i, 2, width, 3);
            }
                

        }
    }
}



__global__ void kernels::calc_d(float* d_d, uchar* d_B_color, int* d_B_texture, unsigned int n_B, float* d_R_color, float* d_R_texture, uchar* d_rgb, int* d_texture, unsigned int height, unsigned int width){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(j < height && i < width){
        float dist_min = 1000;
        float dist;
        for(int k = 0; k < n_B; k++){
            dist = ((l1Dist( &(at4(d_B_color, j, i, k, 0, width, n_B, 3)), &(at3(d_rgb, j, i, 0, width, 3))) / (255.0 * 3) +
        HammingDist( at3(d_B_texture, j, i, k, width, n_B), at2(d_texture, j, i, width)) / (16.0 * 3)) / 2.0);
            if(dist_min > dist){
                dist_min = dist;
            }
        }
        at2(d_d, j, i, width) = dist_min;
            
    }
}

__global__ void kernels::calc_D_min(float* d_D_min, float* d_d, float alpha, unsigned int height, unsigned int width){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(j < height && i < width){
        at2(d_D_min, j, i, width) = at2(d_D_min, j, i, width) * (1 - alpha) + at2(d_d, j, i, width) * alpha;
    }
}

__global__ void kernels::calc_w(float* d_w, bool* d_DR, float* d_Dist_last, float* d_S_feed, unsigned int height, unsigned int width){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(j < height && i < width){
        if(at2(d_DR, j, i, width) == false)
            at2(d_w, j, i, width) = 1.0;
        else if((at2(d_Dist_last, j, i, width) > 0.45) && (at2(d_S_feed, j, i, width) < 0.4))
            at2(d_w, j, i, width) = 1.5;
        else
            at2(d_w, j, i, width) = 0.8;
        

    }
}


__global__ void kernels::calc_v(float* d_v, float v_decr, bool* d_S, bool* d_S_l, float* d_w, unsigned int height, unsigned int width){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(j < height && i < width){
        if(at2(d_S, j, i, width) != at2(d_S_l, j, i, width))
            at2(d_v, j, i, width) = (at2(d_v, j, i, width) + at2(d_w, j, i, width));
        else
            at2(d_v, j, i, width) = (at2(d_v, j, i, width) - v_decr);
        at2(d_v, j, i, width) = clip(at2(d_v, j, i, width), 0.001, 1);
    }
}

__global__ void kernels::update_R(float* d_R, float* d_v, float* d_D_min, unsigned int height, unsigned int width){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(j < height && i < width){
        if(at2(d_R, j, i, width) < pow(1 + at2(d_D_min, j, i, width) * 2, 2))
            at2(d_R, j, i, width) = at2(d_R, j, i, width) + at2(d_v, j, i, width);
        else
            at2(d_R, j, i, width) = at2(d_R, j, i, width) - (1 / at2(d_v, j, i, width));
        at2(d_R, j, i, width) = clip(at2(d_R, j, i, width), 1.0, 4.0);
    }
}

__global__ void kernels::update_T(float* d_T, bool* d_S, float* d_v, float* d_D_min, unsigned int height, unsigned int width){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(j < height && i < width){
        if(at2(d_S, j, i, width) == true)
            at2(d_T, j, i, width) = at2(d_T, j, i, width) + (1.0 / (at2(d_v, j, i, width) * at2(d_D_min, j, i, width)));
        else
            at2(d_T, j, i, width) = at2(d_T, j, i, width) - (at2(d_v, j, i, width) / at2(d_D_min, j, i, width));
        at2(d_T, j, i, width) = clip(at2(d_T, j, i, width), 10, 255);
    }
}


__global__ void kernels::calc_R_color_texture(float* d_R, float* d_R_color, float* d_R_texture, float Ro_color, float Ro_texture, unsigned int height, unsigned int width){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(j < height && i < width){
        at2(d_R_color, j, i, width) = at2(d_R, j, i, width) * Ro_color;
        at2(d_R_texture, j, i, width) = pow(2, at2(d_R, j, i, width)) + Ro_texture;
    }
}


__global__ void kernels::FP_recheck(uchar* d_DB, unsigned int n_DB, bool* d_DR, bool* d_S, uchar* d_rgb, float DB_color_threshold, unsigned int height, unsigned int width){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(j < height && i < width){
        bool b_l1Dist;
        int N = 0;
        if((at2(d_DR, j, i, width) == true) && (at2(d_S, j, i, width) == true)){
            for(int k = 0; k < n_DB; k++){
                b_l1Dist =  (l1Dist( &(at4(d_DB, j, i, k, 0, width, n_DB, 3)), &(at3(d_rgb, j, i, 0, width, 3))) < DB_color_threshold);
                if(b_l1Dist)
                    N++;
            }
            if(N >= 1)
                at2(d_S, j, i, width) = false;

        }
    }
}



// initialization of random states gpu
__global__ void kernels::init_randoms(unsigned int seed, curandState_t* states, int h, int w) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < h & col < w)
      curand_init(seed, row + col, 0, &states[row * w + col]);
  }    
    









/*--------------------------device functions---------------------------*/


__device__ int l1Dist(uchar* B_color, uchar* I_color){
    return abs(B_color[0] - I_color[0]) + abs(B_color[1] - I_color[1]) + abs(B_color[2] - I_color[2]);
}



__device__ int HammingDist(int x, int y){
    int dist = 0;
    char val = x^y;// calculate differ bit
    while(val)   //this dist veriable calculate set bit in loop
    {
        ++dist;
        val &= val - 1;
    }
    return dist;
}

// random numbers from curand states
__device__ int random(curandState_t* states, int h, int w, int j, int i,int min, int max) {
    curandState localState = at2(states, j, i, w);
  
    // generar número pseudoaleatorio
    int ran = min + (curand(&localState)) % (max - min);;
  
    //copiar state de regreso a memoria global
    at2(states, j, i, w) = localState;
  
    //almacenar resultados
    // result[ind] = r;
  
    // printf("%d\n", ran);
  
    return ran;
  }
  

  // clip
__device__ float clip(float i, float a, float b){
    if(i < a)
      i = a;
    if(i > b)
      i = b;
    return i;
  }