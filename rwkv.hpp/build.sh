# create ./build/ directory if not exist

mkdir -p ./build/

# No bf16 support
# g++ -march=skylake-avx512 ./rwkv.cpp -I ./include/ -o ./build/rwkv -g -O3 -std=c++17 -fopenmp  -flto  -fopenmp -funroll-loops -D_GLIBCXX_PARALLEL

# bf16 support (assuming bf16 support is enabled on this machine)
# intel 


# source /opt/intel/oneapi/2024.0/oneapi-vars.sh 
g++ -m64 ./rwkv.cpp -I ./include/ -o ./build/rwkv -march=native -std=c++17 -lvulkan -ffast-math -O3

# COMPILE FOR ARM
# g++ -m64 ./rwkv.cpp -I ./include/ -o ./build/rwkv -mcpu=native -std=c++17 -lvulkan -ffast-math -O3

# g++ -m64 ./rwkv.cpp -I ./include/ -o ./build/rwkv -g -march=native -O3 -std=c++17 -lvulkan

# no avx512 support
# g++ ./rwkv.cpp -I ./include/ -o ./build/rwkv -g -march=haswell -O3 -std=c++17 -fopenmp  -flto  -fopenmp -funroll-loops -D_GLIBCXX_PARALLEL


# copy include/tokenizer/rwkv_vocab_v20230424.txt to ./build/
cp ./include/tokenizer/rwkv_vocab_v20230424.txt ./build/

# run ./build/rwkv
# cd ./build/
# ./rwkv