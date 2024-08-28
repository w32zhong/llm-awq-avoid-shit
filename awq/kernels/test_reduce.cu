#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define N_THREADS 64

__device__ void print(int *debug)
{
    if (threadIdx.x != 0) return;

    if (debug == NULL) {
        for (int i = 0; i < N_THREADS; i ++) {
            printf("%3d ", i);
        }
    } else {
        for (int i = 0; i < N_THREADS; i ++) {
            printf("%3d ", debug[i]);
        }
    }
    printf("\n");
}

__global__ void test(int *init_values)
{
    __shared__ int debug[N_THREADS];
    int value = init_values[threadIdx.x];
    print(NULL);

    debug[threadIdx.x] = value;
    __syncthreads();
    print(debug);

    value += __shfl_xor_sync(~0, value, 16);
    debug[threadIdx.x] = value;
    __syncthreads();
    print(debug);

    value += __shfl_xor_sync(~0, value, 8);
    debug[threadIdx.x] = value;
    __syncthreads();
    print(debug);

    value += __shfl_xor_sync(~0, value, 1);
    debug[threadIdx.x] = value;
    __syncthreads();
    print(debug);

    __syncthreads();
    int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    if (lane == 0 || lane == 2 || lane == 4 || lane == 6)
    {
        printf("out_smem[%d][0/4 + %d] = psum[0/1] \n", warp, lane / 2);
    }
    __syncthreads();
}

int main()
{
    int init_values[N_THREADS];
    for (int i = 0; i < N_THREADS; i ++) {
        init_values[i] = rand() % 10;
    }
    int *d_init_values = NULL;
    cudaMalloc((void**)&d_init_values, sizeof(int) * N_THREADS);
    cudaMemcpy(d_init_values, init_values, sizeof(int) * N_THREADS, cudaMemcpyHostToDevice);

    test<<<1, N_THREADS>>>(d_init_values);

    cudaFree((void*)d_init_values);
}
