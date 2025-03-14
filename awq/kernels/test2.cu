#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>

#define PACK_FACTOR 8
#define WARP_SIZE 32
#define MEM_ACCESS_SIZE 128

// Refer to https://docs.nvidia.com/cuda/archive/12.2.1/pdf/ptx_isa_8.2.pdf
__inline__ __device__ void dequantize_s4_to_fp16x2(half2 const &source, uint4 *result)
{
    uint32_t const i4s = reinterpret_cast<uint32_t const &>(source);
    uint32_t *h = reinterpret_cast<uint32_t *>(result);

    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa; // define operation: (a & b) | c
    static constexpr uint32_t BOTTOM_MASK = 0x000f000f; // two low 4 bits
    static constexpr uint32_t TOP_MASK = 0x00f000f0; // two high 4 bits
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400; // 0x6400 = b0110 0100 0000 0000
    // because fp16 format is (sign 1 bit) (exp 5 bits) (fraction 10 bits)
    // See https://evanw.github.io/float-toy/
    //
    // For any integer 0 ≤ Y < 1024, we can construct the FP16 representation of Y + 1024 by setting the exponent to 1024 and storing Y in the FP16 mantissa. This is easily done by performing 0x6400 | Y , since 0x6400 is the hex representation of 1024 in FP16.
    // -- Who Says Elephants Can’t Run: Bringing Large Scale MoE Models into Cloud Scale Production -- Young Jin Kim

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

    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
    static constexpr uint32_t NEG_64 = 0xd400d400;

    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
}

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

__device__ void decode32(uint32_t x, const char *end)
{
    printf("%hu ", (x >> 0) & 0xffff);
    printf("%hu%s", (x >> 16) & 0xffff, end);
}

/* template: <2, 1, 256, 128> kernel: GridDim=(96,1,1), BlockDim=(256,1,1)*/
template <int NPerBlock, int Batch, int BlockSize, int GroupSize>
__global__ void gemv_kernel(
  const half* inputs, const uint32_t* weight, const half* scales, const half* zeros, half* outputs, 
  const int IC, const int OC) // IC=k, OC=n
{
    const int kStride = 64, kInterleave = 4; /* from pack_intweight() */
    const int kElemsPerThread = MEM_ACCESS_SIZE / 4; // 128 / 4 = 32
    const int kThreadsNumPerTile = kStride / kElemsPerThread; // 64 / 32 = 2

    const int kShuffleBasicTile = 2;
    const int kShuffleContinous = 4;
    const int kShuffleStrided = 4;

    const int Num = NPerBlock * Batch; // 2 * 1 = 2

    half local_inputs[kElemsPerThread];
    uint32_t local_qweights[MEM_ACCESS_SIZE / 32]; // 128 / 32 = 4 x float
    half half_weight_buffer[kElemsPerThread]; 
    half dequantized_weight[kElemsPerThread * NPerBlock]; // 32 * 2
    half local_scale[NPerBlock];
    half local_scaled_zeros[NPerBlock];

    half psum[Num];
    for (int i = 0; i < Num; ++i)
        psum[i] = static_cast<half>(0.f);

    // (cuda-gdb)
    // info cuda kernels
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

    for (int kk = threadIdx.x * kElemsPerThread; kk < IC * kInterleave; kk += BlockSize * kElemsPerThread)
    {
        #pragma unroll
        for (int idx = 0; idx < NPerBlock; ++idx)
        {
            *((float4*)(local_qweights)) = 
                *((float4*)(blk_weight_ptr + (idx * kInterleave * IC + kk)/ PACK_FACTOR));
            local_scale[idx] = *(scale_ptr + idx * kInterleave);
            local_scaled_zeros[idx] = *(zeros_ptr + idx * kInterleave);
            
            #pragma unroll
            for (int i = 0; i < MEM_ACCESS_SIZE / 32; ++i)
            {
                dequantize_s4_to_fp16x2(*reinterpret_cast<half2 *>(local_qweights + i), reinterpret_cast<uint4 *>(half_weight_buffer + i * PACK_FACTOR));

                if (threadIdx.x == 0) {
                    printf("<i = %d> ", i);
                    decode32(local_qweights[i], ": ");
                    for (int j = 0; j < PACK_FACTOR; j++) {
                        printf("%f ", __half2float(half_weight_buffer[i * PACK_FACTOR + j]));
                    }
                    printf("\n");
// <i = 1> 2 3: 2.000000 3.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
// ...
// <i = 1> 40002 40003: 2.000000 3.000000 4.000000 4.000000 12.000000 12.000000 9.000000 9.000000
//
// For
// [40002]
// [40003]
//
// logically 40002 = b 1001 1100 0100 0010 = (9) (12) (4) (2)
// logically 40003 = b 1001 1100 0100 0011 = (9) (12) (4) (3)
// little-indian 40002 = (2) (4) (12) (9)
// little-indian 40003 = (3) (4) (12) (9)
//
// For
// [40002 40010 40018 40026]
// [40003 40011 40019 40027]
// half_weight_buffer[8:16] = 40002 40003 40010 40011 40018 40019 40026 40027
                }
            }

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
                        = __hfma2(
                            *reinterpret_cast<half2*>(dequantized_weight + y * NPerBlock + x * 2),
                            __half2half2(local_inputs[y]),
                            *reinterpret_cast<half2*>(psum + batch_idx * NPerBlock + x * 2)
                        );
                }
            }
        }
        inputs_ptr += act_forward_step;
        scale_ptr += scale_forward_step;
        zeros_ptr += scale_forward_step;
    }

    warp_reduce<Num, WARP_SIZE>(psum, out_smem);
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

void fill_kernel(uint16_t *kernel, const int n, const int k)
{
    FILE *fh = fopen("test2.json", "r");
    uint16_t x;
    int cnt = 0, res = 1;
    while (1) {
        res = fscanf(fh, "%hu,", &x);
        if (res <= 0) break;

        if (cnt % 4 == 0) {
            kernel[cnt / 4] = x;
        }
        //kernel[cnt / 4] |= (
        //    x << (4 * (cnt % 4))
        //);
        cnt ++;
    }

    printf("fill_kernel: %d numel.\n", cnt);
    for (int i = 0; i < n/4; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%hu \t", kernel[i * k + j]);
        }
        printf("\n");
    }
}

int main()
{
    const int m = 1;
    const int n = 8; // 768; // out_features
    const int k = 384; // 3072; // in_features
    /* (m, k) @ (k, n) */

    static constexpr int N_PER_BLOCK = 2;
    static constexpr int K_INTERLEAVE = 4;
    static constexpr int BLOCK_SIZE = 256;

    const half *in_feats = NULL;
    cudaMalloc((void **)&in_feats, sizeof(half) * m * 1 * k);

    size_t kernel_size = sizeof(uint16_t) * (n / K_INTERLEAVE) * k;
    uint16_t *h_kernel = (uint16_t*)malloc(kernel_size);
    fill_kernel(h_kernel, n, k);

    uint32_t *kernel = NULL;
    cudaMalloc((void **)&kernel, sizeof(uint16_t) * (n / K_INTERLEAVE) * k);
    cudaMemcpy(kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice);

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

    free(h_kernel);
    cudaFree((void*)in_feats);
    cudaFree((void*)kernel);
    cudaFree((void*)zeros);
    cudaFree((void*)scaling_factors);
    cudaFree((void*)out_feats);
    return 0;
}
