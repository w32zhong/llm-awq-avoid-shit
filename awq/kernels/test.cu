#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>

#define PACK_FACTOR 8
#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float sum) {
  #pragma unroll
  for(int i = 4; i >= 0; i--){
    sum += __shfl_down_sync(0xffffffff, sum, 1<<i);
  }
  /*
  // Equivalent to the following tree reduction implementation:
  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);
  */
  return sum;
}

__device__ __forceinline__ int make_divisible(int c, int divisor){
  return (c + divisor - 1) / divisor;
}


__global__ void gemv_kernel_g128(
  const float4* _inputs, const uint32_t* weight, const uint32_t* zeros, const half* scaling_factors, half* _outputs, 
  const int IC, const int OC){

    assert(blockDim.x == 32);
    assert(blockDim.y == 4);
    assert(blockDim.z == 1);
    assert(blockIdx.x == 0);
    assert(blockIdx.y < 192);
    assert(blockIdx.z == 0);
    assert(threadIdx.x < 32);
    assert(threadIdx.y < 4);
    assert(threadIdx.z == 0);
    if (blockIdx.y == 191 && threadIdx.x == 31 && threadIdx.y == 3) {
        printf("case study\n");
    }
    // cuda-gdb:
    // cuda device sm warp lane
    // cuda kernel block thread
    // cuda block (0, 190, 0) # switch block
    // cuda thread (30, 2, 0) # switch thread
    // info cuda threads # show all threads and the currently active thread
    // info cuda kernels
    // info cuda blocks
    // info cuda warps
    // frame

    const int group_size = 128;
    float psum = 0;
    const int batch_idx = blockIdx.z;
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 

    const float4* inputs = _inputs + batch_idx * IC / PACK_FACTOR;
    half* outputs = _outputs + batch_idx * OC;

    const int weight_w = IC / PACK_FACTOR;
    const int zeros_w = make_divisible(IC / group_size, PACK_FACTOR);
    // scaling_factor width
    const int sf_w = make_divisible(IC / group_size, PACK_FACTOR) * PACK_FACTOR;

    // tile size: 4 OC x 1024 IC per iter
    const int num_groups_packed = make_divisible(IC / group_size, PACK_FACTOR);
    for(int packed_group_idx = 0; packed_group_idx < num_groups_packed; packed_group_idx++) {

      uint32_t packed_weights[4];
      *((float4*)(packed_weights)) = *((float4*)(weight + oc_idx * weight_w + packed_group_idx * (WARP_SIZE * 4) + threadIdx.x * 4));

      float scaling_factor = __half2float(scaling_factors[oc_idx * sf_w + packed_group_idx * 8 + (threadIdx.x / 4)]);

      uint32_t packed_zeros = *(zeros + oc_idx * zeros_w + packed_group_idx);
      float current_zeros = (float)((
        packed_zeros >> (threadIdx.x / 4 * 4) /* 0, 4, 8, ..., 32 */
        ) & 0xF);

      int inputs_ptr_delta = packed_group_idx * WARP_SIZE * 4 + threadIdx.x * 4; /* 0, 4, 8, ..., (32 * 4 = 128). */
      const float4* inputs_ptr = inputs + inputs_ptr_delta; /* inputs is of float4, each time load 4 float4! */

      #pragma unroll
      for (int ic_0 = 0; ic_0 < 4; ic_0++) {
        uint32_t current_packed_weight = packed_weights[ic_0];
        half packed_inputs[PACK_FACTOR]; /* 8 x half = 16 bytes = float4 */
        if (inputs_ptr_delta + ic_0 < IC / PACK_FACTOR) {
          *((float4*)packed_inputs) = *(inputs_ptr + ic_0); /* load 1 out of 4 float4! */
          #pragma unroll
          for (int ic_1 = 0; ic_1 < PACK_FACTOR; ic_1++){
            float current_single_weight_fp = (float)(current_packed_weight & 0xF);
            float dequantized_weight = scaling_factor * (current_single_weight_fp - current_zeros);
            psum += dequantized_weight * __half2float(packed_inputs[ic_1]);
            current_packed_weight = current_packed_weight >> 4; /* shift */
          } /* end inner inner for */
        } /* end if guard */
      } /* end inner for */
    } /* end outter for */

    psum = warp_reduce_sum(psum);
    if (threadIdx.x == 0) {
        outputs[oc_idx] = __float2half(psum); 
    }
}

int main()
{
    const float4 *in_feats = NULL;
    cudaMalloc((void **)&in_feats, sizeof(float) * 1 * 768);
    //int num_in_feats = 1;
    int num_in_channels = 768;

    const uint32_t *kernel = NULL;
    cudaMalloc((void **)&kernel, sizeof(uint32_t) * 768 * 96);

    const uint32_t *zeros = NULL;
    cudaMalloc((void **)&zeros, sizeof(uint32_t) * 768 * 1);

    const half *scaling_factors = NULL;
    cudaMalloc((void **)&scaling_factors, sizeof(float) * 768 * 8);

    half *out_feats = NULL;
    cudaMalloc((void **)&out_feats, sizeof(float) * 1 * 768);
    int num_out_feats = 1;
    int num_out_channels = 768;

    dim3 num_blocks(1, num_out_channels / 4, num_out_feats);
    dim3 num_threads(32, 4); // blockDim: (32, 4, 1)
    {
      gemv_kernel_g128<<<num_blocks, num_threads>>>(
        in_feats, kernel, zeros, scaling_factors, out_feats,
        num_in_channels, num_out_channels
      );
    }

    cudaFree((void*)in_feats);
    cudaFree((void*)kernel);
    cudaFree((void*)zeros);
    cudaFree((void*)scaling_factors);
    cudaFree((void*)out_feats);
    return 0;
}
