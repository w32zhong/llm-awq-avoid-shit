import math
import torch
import torch.nn as nn
import awq_inference_engine  # with CUDA kernels


def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError
    
    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width


# added for the new quantization kernel
def pack_intweight(unpacked_qweight, interleave, kstride):
    # unpacked_qweight: [N, K]
    N = unpacked_qweight.shape[0] # 768
    K = unpacked_qweight.shape[1] # 768

    # 768 / 4 = 192
    #unpacked_qweight[0] = torch.arange(768)
    #unpacked_qweight[1] = torch.arange(768) + 1000
    #unpacked_qweight[2] = torch.arange(768) + 2000
    #unpacked_qweight[3] = torch.arange(768) + 3000

    # step1 = np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2).reshape(32)
    Packed_Kernel = unpacked_qweight.cpu().numpy().reshape(N, K // 32, 32)
    save1 = Packed_Kernel
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 32)
    # step1 [ 0,  1,  8,  9, 16, 17, 24, 25,  2,  3, 10, 11, 18, 19, 26, 27,  4,
    #         5, 12, 13, 20, 21, 28, 29,  6,  7, 14, 15, 22, 23, 30, 31]

    # step2 = step1.reshape(4, 4, 2).transpose(0, 2, 1).reshape(32)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 8)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    Packed_Kernel = Packed_Kernel.reshape(N, K)
    save2 = Packed_Kernel.reshape(N, K // 32, 32)
    # step2 [ 0,  8, 16, 24,  1,  9, 17, 25,  2, 10, 18, 26,  3, 11, 19, 27,  4,
    #        12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31])

    # save2[0,0] == save1[0,0][[0, 8, 16, 24, 1, 9, 17, 25, ...]]

    # interleave = 4, kstride = 64
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, interleave, K // kstride, kstride
        #           192,          4,           12,      64
    )
    Packed_Kernel = Packed_Kernel.transpose(0, 2, 1, 3) # 192, 12, 4, 64
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, K // kstride, kstride, interleave
    ) # 192, 12, 64, 4
    save3 = Packed_Kernel
    
    # Packed_Kernel[0,0] == save2[0,0], save2[0,1] ...
    # p save3[0][0]
    # p save3[0][1]
    # ...

    Packed_Kernel = (
        Packed_Kernel[..., 0]
        | (Packed_Kernel[..., 1] << 4)
        | (Packed_Kernel[..., 2] << 8)
        | (Packed_Kernel[..., 3] << 12)
    ) # 192, 12, 64

    Packed_Kernel = Packed_Kernel.reshape(N // interleave, K) # 192, 768
    qweight = (
        torch.tensor(Packed_Kernel.astype("int16"))
        .to(unpacked_qweight.device)
        .contiguous()
    )
    return qweight
    

class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)
    
    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


class WQLinear(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()
        
        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.split_k_iters = 8
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        pack_num = (32 // self.w_bit)
        # TODO (Haotian): a function for buffer shape calculation
        self.register_buffer('qweight', torch.zeros((out_features, in_features // pack_num), dtype=torch.int32, device=dev))
        self.register_buffer('qzeros', torch.zeros((out_features, calculate_zeros_width(in_features, self.group_size)), dtype=torch.int32, device=dev))
        self.register_buffer('scales', torch.zeros((out_features, calculate_zeros_width(in_features, self.group_size) * pack_num), dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None):
        awq_linear = cls(w_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device)
        if init_only:  # just prepare for loading sd
            return awq_linear
        
        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None  
        scale_zeros = zeros * scales

        pack_num = 32 // awq_linear.w_bit
        qscales = torch.zeros(
            (scales.shape[0], calculate_zeros_width(linear.in_features, group_size) * pack_num),
            dtype=torch.float16,
            device=scales.device
        )
        qscales[:, :scales.shape[1]] = scales
        # awq_linear.scales = scales.clone().half()
        awq_linear.scales = qscales
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()
        
        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(torch.round((linear.weight.data[:, idx] + scale_zeros[:, idx // group_size]) / awq_linear.scales[:, idx // group_size]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        # intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)
        awq_linear.intweight = intweight

        qweight = torch.zeros((intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit), dtype=torch.int32, device=intweight.device)           
         
        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                # order_map = [0, 2, 4, 6, 1, 3, 5, 7]
                order_map = [0, 1, 2, 3, 4, 5, 6, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight

        awq_linear.qweight2 = pack_intweight(
            intweight.contiguous(), interleave=4, kstride=64
        )

        zeros = zeros.to(dtype=torch.int32)
        qzeros = torch.zeros(
            (zeros.shape[0], calculate_zeros_width(linear.in_features, group_size)),
            dtype=torch.int32,
            device=zeros.device,
        )
        
        for col in range((zeros.shape[1] + pack_num - 1) // pack_num):
            if awq_linear.w_bit == 4:
                # order_map = [0, 2, 4, 6, 1, 3, 5, 7]
                order_map = [0, 1, 2, 3, 4, 5, 6, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                if col * pack_num + order_map[i] >= zeros.shape[1]:
                    continue
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros

        # added for the new quantization kernel
        scaled_zeros = torch.zeros_like(qscales)
        scaled_zeros[:, : scales.shape[1]] = -(
            qscales[:, : scales.shape[1]] * (zeros.to(torch.float32))
        ).to(torch.float16)
        awq_linear.scaled_zeros = scaled_zeros

        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        inputs = x.reshape(-1, x.shape[-1])
        #print(self.path, inputs.shape)
        if inputs.shape[0] >= 8:
            out = awq_inference_engine.gemm_forward_cuda(inputs, self.qweight, self.scales, self.qzeros, self.group_size, self.split_k_iters)
        else:
            #out = awq_inference_engine.gemv_forward_cuda(inputs, self.qweight, self.scales, self.qzeros, self.group_size)

            inputs = inputs.unsqueeze(0).contiguous()
            qweight = self.qweight2
            scales = self.scales.transpose(0, 1).contiguous()
            scaled_zeros = self.scaled_zeros.transpose(0, 1).contiguous()
            out = awq_inference_engine.gemv_forward_cuda_new(
                inputs, qweight, scales, scaled_zeros,
                inputs.numel() // inputs.shape[-1],
                self.out_features,
                self.in_features,
                self.group_size
            ) # out.shape: [1, 1, 768]
            #breakpoint()
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, w_bit={}, group_size={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.w_bit, self.group_size
        )
