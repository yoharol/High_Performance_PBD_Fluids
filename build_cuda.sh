export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.9/compilers/bin:$PATH
nvcc pbf.cu -o build/pbf_cuda
./build/pbf_cuda 32
python3 post_run.py cuda
