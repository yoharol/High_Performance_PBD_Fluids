clang++ -O3 -mavx512f -mfma pbf_simd.cpp -o build/pbf_simd
./build/pbf_simd simd
python3 post_run.py simd