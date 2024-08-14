set -xe

nvcc \
    -I/home/tk/anaconda3/envs/$CONDA_DEFAULT_ENV/lib/python3.9/site-packages/torch/include \
    -I/home/tk/anaconda3/envs/$CONDA_DEFAULT_ENV/lib/python3.9/site-packages/torch/include/torch/csrc/api/include \
    -I/home/tk/anaconda3/envs/$CONDA_DEFAULT_ENV/include/python3.9 \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -c test.cu -g -O0 -std=c++17 

$CXX -L/home/tk/anaconda3/envs/$CONDA_DEFAULT_ENV/lib/python3.9/site-packages/torch/lib \
    -Wl,-rpath,/home/tk/anaconda3/envs/$CONDA_DEFAULT_ENV/lib/python3.9/site-packages/torch/lib \
    test.o -lc10 -ltorch_cpu -lcudart

#export LD_LIBRARY_PATH=/home/tk/anaconda3/envs/$CONDA_DEFAULT_ENV/lib/python3.9/site-packages/torch/lib
./a.out
