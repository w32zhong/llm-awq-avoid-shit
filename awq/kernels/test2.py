def pack_matrix(unpacked_qweight, interleave=4, kstride=64):
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
    #Packed_Kernel = (
    #    Packed_Kernel[..., 0]
    #    | (Packed_Kernel[..., 1] << 4)
    #    | (Packed_Kernel[..., 2] << 8)
    #    | (Packed_Kernel[..., 3] << 12)
    #)


import numpy as np
m, n = 8, 128
A = np.zeros((m, n), dtype='int')
for row in range(A.shape[0]):
    A[row] = np.arange(n) + 1000 * row
print(A.shape)
print(A)
np.set_printoptions(threshold=np.inf)
Packed_A = pack_matrix(A)
print(Packed_A.shape)
print(Packed_A)
