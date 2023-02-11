clang++ -O3 pbf_baseline.cpp -o build/pbf_baseline
./build/pbf_baseline baseline
python3 post_run.py baseline
