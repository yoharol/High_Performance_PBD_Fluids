clang++ -O3 -mavx512f -mfma -fopenmp pbf_omp.cpp -o build/pbf_omp
./build/pbf_omp omp
python3 post_run.py omp

OMP_NUM_THREADS=16 ./build/pbf_omp omp16
python3 post_run.py opm16