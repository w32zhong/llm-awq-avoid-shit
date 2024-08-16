kStride = 64
kInterleave = 4

def pack_matrix(unpacked_qweight, interleave=kInterleave, kstride=kStride):
    N = unpacked_qweight.shape[0]
    K = unpacked_qweight.shape[1]

    Packed_Kernel = unpacked_qweight.reshape(N, K // 32, 32)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 32)

    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 8)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    Packed_Kernel = Packed_Kernel.reshape(N, K)

    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, interleave, K // kstride, kstride
    )
    Packed_Kernel = Packed_Kernel.transpose(0, 2, 1, 3)
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, K // kstride, kstride, interleave
    )

    return Packed_Kernel.reshape(N // interleave, K, interleave)


import numpy as np
Batch = 1
n, k = 8, 384 # 3 groups
IC, OC = k, n
A = np.zeros((n, k), dtype='int')
for row in range(n):
    A[row] = np.arange(k) + 10000 * row
print(A.shape)
print(A)
Packed_A = pack_matrix(A)
NPerBlock = 2
K_INTERLEAVE = 4
BlockSize = 256
blockRange = range(n // NPerBlock // K_INTERLEAVE)
threadRange = range(BlockSize)
print('blockIdx:', blockRange)
print('threadIdx:', threadRange)

GroupSize = 128
WARP_SIZE = 32
PACK_FACTOR = 8
MEM_ACCESS_SIZE = 128
kElemsPerThread = MEM_ACCESS_SIZE // 4
kThreadsNumPerTile = kStride // kElemsPerThread

kShuffleBasicTile = 2
kShuffleContinous = 4
kShuffleStrided = 4

print(f'local_inputs[{kElemsPerThread}]')
print(f'half_weight_buffer[{kElemsPerThread}]')
print(f'dequantized_weight[{kElemsPerThread * NPerBlock}]')
print(f'local_scale[{NPerBlock}]')

def get_float4(matrix, offset):
    return matrix.flat[offset: offset + 8 * 4]

def get_xy(offset, width):
    x = offset // width
    y = offset % width
    return f'({x}, {y})'

def gemv_kernel(matrix, blockIdx, threadIdx):
    blk_row_offset = blockIdx * NPerBlock * kInterleave
    thd_row_offset = (threadIdx // kThreadsNumPerTile) % kInterleave
    act_k_offset = threadIdx // (kThreadsNumPerTile * kInterleave) * kStride + (threadIdx % kThreadsNumPerTile) * kElemsPerThread
    group_offset = act_k_offset // GroupSize
    blk_weight_ptr = 8 * (blk_row_offset * IC // PACK_FACTOR)
    scale_ptr = blk_row_offset + thd_row_offset + group_offset * OC
    inputs_ptr = act_k_offset
    act_forward_step = BlockSize * kElemsPerThread // kInterleave
    scale_forward_step = act_forward_step // GroupSize * OC

    kkRange = range(threadIdx * kElemsPerThread, IC * kInterleave, BlockSize * kElemsPerThread)
    for idx_kk, kk in enumerate(kkRange):
        assert idx_kk == 0
        for idx in range(NPerBlock):
            offset = (idx * kInterleave * IC + kk) // PACK_FACTOR
            print(f'local_qweights[0:4] = matrix{get_float4(matrix, blk_weight_ptr + 8 * offset)}')
            print(f'local_scale[{idx}] = scale{get_xy(scale_ptr + idx * kInterleave, OC)}')
            matrix.flat[blk_weight_ptr + 8 * offset] = -1

            for i in range(MEM_ACCESS_SIZE // 32):
                print(f'dequantize_s4_to_fp16x2(local_qweights[{i}], half_weight_buffer[{i * PACK_FACTOR}])')
            for i in range(kShuffleContinous):
                for j in range(kShuffleStrided):
                    offset = (i + j * kShuffleContinous) * kShuffleBasicTile
                    print(f'w = *(half2*)half_weight_buffer[{offset}]')
                    print(f'w = w * local_scale[{idx}] + zeros[{idx}]')
                    offset = ((i * kShuffleStrided + j) * kShuffleBasicTile + 0) * NPerBlock + idx
                    print(f'dequantized_weight[{offset}] = w.x')
                    offset = ((i * kShuffleStrided + j) * kShuffleBasicTile + 1) * NPerBlock + idx
                    print(f'dequantized_weight[{offset}] = w.y')

        for batch_idx in range(1):
            for idx in range(kElemsPerThread // 8):
                print(f'local_inputs[{idx * 8}] = inputs[{inputs_ptr} + {idx * 8}]')
                
            for x in range(NPerBlock // 2):
                for y in range(kElemsPerThread):
                    sum_off = x * 2
                    dw_off = y * NPerBlock + x * 2
                    print(f'psum[{sum_off}] += dequantized_weight[{dw_off}] * local_inputs[{y}]')

    for i in range(threadIdx, NPerBlock * kInterleave, BlockSize):
        batch_idx = i // (NPerBlock * kInterleave);
        oc_idx = i % (NPerBlock * kInterleave);
        for j in range(BlockSize // WARP_SIZE):
            print(f'warp sum += out_smem[{j}][{i}]')
        offset = batch_idx * OC + blk_row_offset + oc_idx
        print(f'threadIdx={threadIdx}: outputs[{offset}] = warp sum')


if __name__ == '__main__':
    for blockIdx in blockRange:
        for threadIdx in threadRange:
            gemv_kernel(Packed_A, blockIdx, threadIdx)
    np.set_printoptions(threshold=np.inf)
    print(Packed_A.shape)
    print(Packed_A)
