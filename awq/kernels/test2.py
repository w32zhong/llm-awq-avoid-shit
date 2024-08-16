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
n, k = 8, 384 # 3 groups
IC, OC = k, n
A = np.zeros((n, k), dtype='int')
for row in range(n):
    A[row] = np.arange(k) + 10000 * row
print(A.shape)
print(A)
np.set_printoptions(threshold=np.inf)
Packed_A = pack_matrix(A)
print(Packed_A.shape)

NPerBlock = 2
K_INTERLEAVE = 4
BlockSize = 256
blockRange = range(n // NPerBlock // K_INTERLEAVE)
threadRange = range(BlockSize)
print('blockIdx:', blockRange)
print('threadIdx:', threadRange)

GroupSize = 128
PACK_FACTOR = 8
MEM_ACCESS_SIZE = 128
kElemsPerThread = MEM_ACCESS_SIZE // 4
kThreadsNumPerTile = kStride // kElemsPerThread

def get_float4(matrix, offset):
    return matrix.flat[offset: offset + 8 * 4]

def get_xy(offset, width):
    x = offset // width
    y = offset % width
    return f'({x}, {y})'

def process(matrix, blockIdx, threadIdx):
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
    for kk in kkRange:
        for idx in range(0, NPerBlock):
            offset = (idx * kInterleave * IC + kk) // PACK_FACTOR
            print(f'local_qweights[0,1,2,3] = matrix{get_float4(matrix, blk_weight_ptr + 8 * offset)}')
            print(f'local_scale[{idx}] = scale{get_xy(scale_ptr + idx * kInterleave, OC)}')
            matrix.flat[blk_weight_ptr + 8 * offset] = -1

for blockIdx in blockRange:
    for threadIdx in threadRange:
        process(Packed_A, blockIdx, threadIdx)
print(Packed_A)
