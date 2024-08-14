set -x
nvcc \
    -I/home/tk/anaconda3/envs/awq-avoid-shit/lib/python3.9/site-packages/torch/include \
    -I/home/tk/anaconda3/envs/awq-avoid-shit/lib/python3.9/site-packages/torch/include/torch/csrc/api/include \
    -I/home/tk/anaconda3/envs/awq-avoid-shit/include/python3.9 \
    -c test.cu -o test.out -g -O0 -std=c++17 -ccbin $CC
