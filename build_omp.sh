clang++ -O3 -mavx512f -mfma -fopenmp pbf_omp.cpp -o build/pbf_omp
./build/pbf_omp omp
python3 post_run.py omp