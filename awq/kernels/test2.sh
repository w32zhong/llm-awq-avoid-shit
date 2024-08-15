set -xe

nvcc \
    -I/home/tk/anaconda3/envs/$CONDA_DEFAULT_ENV/lib/python3.9/site-packages/torch/include \
    -I/home/tk/anaconda3/envs/$CONDA_DEFAULT_ENV/lib/python3.9/site-packages/torch/include/torch/csrc/api/include \
    -I/home/tk/anaconda3/envs/$CONDA_DEFAULT_ENV/include/python3.9 \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -gencode=arch=compute_86,code=sm_86 \
    -c test2.cu -g -G -O0 -std=c++17 #--dryrun

$CXX -L/home/tk/anaconda3/envs/$CONDA_DEFAULT_ENV/lib/python3.9/site-packages/torch/lib \
    -Wl,-rpath,/home/tk/anaconda3/envs/$CONDA_DEFAULT_ENV/lib/python3.9/site-packages/torch/lib \
    test2.o -lc10 -ltorch_cpu -lcudart

#export LD_LIBRARY_PATH=/home/tk/anaconda3/envs/$CONDA_DEFAULT_ENV/lib/python3.9/site-packages/torch/lib
./a.out
