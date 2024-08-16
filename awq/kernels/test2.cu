#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>

#define PACK_FACTOR 8
#define WARP_SIZE 32
#define MEM_ACCESS_SIZE 128

__inline__ __device__ void dequantize_s4_to_fp16x2(half2 const &source, uint4 *result)
{
    // uint4 result;

    uint32_t *h = reinterpret_cast<uint32_t *>(result);
    uint32_t const i4s = reinterpret_cast<uint32_t const &>(source);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
    static constexpr uint32_t TOP_MASK = 0x00f000f0;
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

    // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
    // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
    // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
    // elt_67 to fp16 without having to shift them to the bottom bits before hand.

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
    // immediately before required.
    const uint32_t top_i4s = i4s >> 8;
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[0])
                 : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[1])
                 : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[2])
                 : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[3])
                 : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    // I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
    // half2 ctor. In this case, I chose performance reliability over code readability.

    // This is the half2 {1032, 1032} represented as an integer.
    // static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
    // Haotian: subtract {1024, 1024} instead, we do not need to map to [-8, 7]
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
    // This is the half2 {1 / 16, 1 / 16} represented as an integer.
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
    // This is the half2 {-72, -72} represented as an integer.
    // static constexpr uint32_t NEG_72 = 0xd480d480;
    // Haotian: Let's use {-64, -64}.
    static constexpr uint32_t NEG_64 = 0xd400d400;

    // Finally, we construct the output numbers.
    // Convert elt_01
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_23
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
    // Convert elt_45
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_67
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

    // return result;
}

// Reduce sum within the warp using the tree reduction algorithm.
template <int Num, int WarpSize>
__device__ __forceinline__ static void warp_reduce(half* psum, float (*out_smem)[Num * 4])
{
  // kInterleave = 4
      float fpsum[Num];
      #pragma unroll
      for (int i = 0; i < Num; ++i)
      {
          fpsum[i] = static_cast<float>(psum[i]);
      }

      #pragma unroll
      for (int i = 0; i < Num; ++i)
      {
          // T0 + T1 + T8 + T9 + T16 + T17 + T24 + T25 (kInterleave = 4)
          fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 16);
          fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 8);
          fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 1);
      }
      __syncthreads();
      int warp = threadIdx.x / WarpSize, lane = threadIdx.x % WarpSize;
      if (lane == 0 || lane == 2 || lane == 4 || lane == 6)
      {
          #pragma unroll
          for (int i = 0; i < Num; ++i)
          {
              out_smem[warp][i * 4 + lane / 2] = fpsum[i];
          }
      }
      __syncthreads();
};

/* template: <2, 1, 256, 128> kernel: GridDim=(96,1,1), BlockDim=(256,1,1)*/
template <int NPerBlock, int Batch, int BlockSize, int GroupSize>
__global__ void gemv_kernel(
  const half* inputs, const uint32_t* weight, const half* scales, const half* zeros, half* outputs, 
  const int IC, const int OC) // IC == OC == 768
{
    const int kStride = 64, kInterleave = 4; /* from pack_intweight() */
    const int kElemsPerThread = MEM_ACCESS_SIZE / 4; // 128 / 4 = 32
    const int kThreadsNumPerTile = kStride / kElemsPerThread; // 64 / 32 = 2

    const int kShuffleBasicTile = 2;
    const int kShuffleContinous = 4;
    const int kShuffleStrided = 4;

    const int Num = NPerBlock * Batch; // 2 * 1 = 2

    half local_inputs[kElemsPerThread];
    uint32_t local_qweights[MEM_ACCESS_SIZE / 32];
    half half_weight_buffer[kElemsPerThread]; 
    half dequantized_weight[kElemsPerThread * NPerBlock];
    half local_scale[NPerBlock];
    half local_scaled_zeros[NPerBlock];

    half psum[Num];
    for (int i = 0; i < Num; ++i)
        psum[i] = static_cast<half>(0.f);

    // (cuda-gdb)
    // info cuda threads
    // cuda block (55,0,0)
    // cuda thread (31,0,0)
    
    extern __shared__ uint8_t shmem[];
    float(*out_smem)[Num * kInterleave] = reinterpret_cast<float(*)[Num * kInterleave]>(shmem);

    const int blk_row_offset = blockIdx.x * NPerBlock * kInterleave;
    const int thd_row_offset = (threadIdx.x / kThreadsNumPerTile) % kInterleave;
    const int act_k_offset = threadIdx.x / (kThreadsNumPerTile * kInterleave) * kStride
                               + (threadIdx.x % kThreadsNumPerTile) * kElemsPerThread;
    const int group_offset = act_k_offset / GroupSize;
    const uint32_t* blk_weight_ptr = weight + blk_row_offset * IC / PACK_FACTOR;
    const half* scale_ptr = scales + blk_row_offset + thd_row_offset + group_offset * OC;
    const half* zeros_ptr = zeros + blk_row_offset + thd_row_offset + group_offset * OC;
    const half* inputs_ptr = inputs + act_k_offset;

    const int act_forward_step = BlockSize * kElemsPerThread / kInterleave;
    const int scale_forward_step = act_forward_step / GroupSize * OC;

    // Main loop iteration, each block completes the outputs for several OCs
    for (int kk = threadIdx.x * kElemsPerThread; kk < IC * kInterleave; kk += BlockSize * kElemsPerThread)
    {
        // Load qweight, scales and scaled_zeros
        #pragma unroll
        for (int idx = 0; idx < NPerBlock; ++idx)
        {
            // use float4 to load weights, each thread load 32 int4 numbers (1 x float4, 128 bit)
            *((float4*)(local_qweights)) = 
                *((float4*)(blk_weight_ptr + (idx * kInterleave * IC + kk)/ PACK_FACTOR));
            local_scale[idx] = *(scale_ptr + idx * kInterleave);
            local_scaled_zeros[idx] = *(zeros_ptr + idx * kInterleave);
            
            // Map int4 qweight to fp format 
            #pragma unroll
            for (int i = 0; i < MEM_ACCESS_SIZE / 32; ++i)
            {
                // Converts 32 bits (8 x int4) to 8 fp16
                dequantize_s4_to_fp16x2(*reinterpret_cast<half2 *>(local_qweights + i), reinterpret_cast<uint4 *>(half_weight_buffer + i * PACK_FACTOR));
            }

            // Dequantize (apply s/z) and shuffle elements to match the weight packing format
            #pragma unroll
            for (int i = 0; i < kShuffleContinous; ++i)
            {
                #pragma unroll
                for (int j = 0; j < kShuffleStrided; ++j)
                {
                    half2 w = 
                        *reinterpret_cast<half2*>(
                          half_weight_buffer + (i + j * kShuffleContinous)* kShuffleBasicTile
                        );
                    w = __hfma2(w, __half2half2(local_scale[idx]), __half2half2(local_scaled_zeros[idx]));
                    dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 0) 
                          * NPerBlock + idx]
                        = w.x;
                    dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 1)
                            * NPerBlock + idx]
                        = w.y;
                }
            }            
        }  
        #pragma unroll
        for (int batch_idx = 0; batch_idx < Batch; ++batch_idx)
        {
            const half* local_inputs_ptr = inputs_ptr + batch_idx * IC;
            #pragma unroll
            for (int idx = 0; idx < kElemsPerThread / 8; ++idx)
            {
                // load activation, 8 halves (128 bits) / step.
                *((float4*)(local_inputs + idx * 8)) = *((float4*)(local_inputs_ptr + idx * 8));
            }
            // Perform the MACs
            #pragma unroll
            for (int x = 0; x < NPerBlock / 2; ++x)
            {
                #pragma unroll
                for (int y = 0; y < kElemsPerThread; ++y)
                {
                    *reinterpret_cast<half2*>(psum + batch_idx * NPerBlock + x * 2)
                        = __hfma2(*reinterpret_cast<half2*>(dequantized_weight + y * NPerBlock + x * 2),
                            __half2half2(local_inputs[y]),
                            *reinterpret_cast<half2*>(psum + batch_idx * NPerBlock + x * 2));
                }
            }
        }
        inputs_ptr += act_forward_step;
        scale_ptr += scale_forward_step;
        zeros_ptr += scale_forward_step;
    }

    warp_reduce<Num, WARP_SIZE>(psum, out_smem);

    // Num * Interleave = batch * NPerBlock * Interleave -> 1 thread_block write back num
    for (int i = threadIdx.x; i < Num * kInterleave; i += BlockSize)
    {
        int batch_idx = i / (NPerBlock * kInterleave);
        int oc_idx = i % (NPerBlock * kInterleave);
        float acc = 0.f;
        for (int j = 0; j < BlockSize / WARP_SIZE; ++j)
        {
            acc += out_smem[j][i];
        }
        outputs[batch_idx * OC + blk_row_offset + oc_idx] = static_cast<half>(acc);
    }
}

int main()
{
    const int m = 1;
    const int n = 768; // out_features
    const int k = 3072; // in_features
    /* (m, k) @ (k, n) */

    static constexpr int N_PER_BLOCK = 2;
    static constexpr int K_INTERLEAVE = 4;
    static constexpr int BLOCK_SIZE = 256;

    const half *in_feats = NULL;
    cudaMalloc((void **)&in_feats, sizeof(half) * m * 1 * k);

    const uint32_t *kernel = NULL;
    cudaMalloc((void **)&kernel, sizeof(uint16_t) * (n / K_INTERLEAVE) * k);

    const half *scaling_factors = NULL;
    cudaMalloc((void **)&scaling_factors, sizeof(half) * (k / 128) * n);

    const half *zeros = NULL;
    cudaMalloc((void **)&zeros, sizeof(half) * (k / 128) * n);

    half *out_feats = NULL;
    cudaMalloc((void **)&out_feats, sizeof(half) * m * 1 * n);

    dim3 num_blocks(n / N_PER_BLOCK / K_INTERLEAVE);
    dim3 num_threads(BLOCK_SIZE);
    {
        gemv_kernel<N_PER_BLOCK, m, BLOCK_SIZE, 128><<<num_blocks, num_threads>>>(
          in_feats, kernel, scaling_factors, zeros, out_feats, k, n
        );
    }

    cudaFree((void*)in_feats);
    cudaFree((void*)kernel);
    cudaFree((void*)zeros);
    cudaFree((void*)scaling_factors);
    cudaFree((void*)out_feats);
    return 0;
}
