# create ./build/ directory if not exist

mkdir -p ./build/

# No bf16 support
# g++ -march=skylake-avx512 ./rwkv.cpp -I ./include/ -o ./build/rwkv -g -O3 -std=c++17 -fopenmp  -flto  -fopenmp -funroll-loops -D_GLIBCXX_PARALLEL

# bf16 support (assuming bf16 support is enabled on this machine)
# intel 

# check if vulkan installed
vulk=$(pkg-config --cflags --libs vulkan)

# set vulk to empty string if vulkan not installed or ld -lvulkan fails
if [ -z "$vulk" ]; then
    echo "Vulkan not installed"
    vulk=""
fi



# build with intel compiler
# check if intel compiler installed
if [ "$(icpx --version)" ]; then
icpx -m64 ./rwkv.cpp -I ./include/ -o ./build/rwkv -march=native -std=c++17 $vulk -ffast-math -O3 -pthread

else
# build with g++
#check if g++ installed, use clang++ if g++ not installed
if [ -z "$(g++ --version)" ]; then
    echo "g++ not installed, using clang++"
    clang++ -m64 ./rwkv.cpp -I ./include/ -o ./build/rwkv -march=native -std=c++17 $vulk -ffast-math -O3 -pthread
else
    g++ -m64 ./rwkv.cpp -I ./include/ -o ./build/rwkv -march=native -std=c++17 $vulk -ffast-math -O3 -pthread
fi
fi
# g++ -m64 ./rwkv.cpp -I ./include/ -o ./build/rwkv -march=native -std=c++17 $vulk -ffast-math -O3 -pthread

# build with clang++
# clang++ -m64 ./rwkv.cpp -I ./include/ -o ./build/rwkv -march=native -std=c++17 $vulk -ffast-math -O3 -pthread

# COMPILE FOR ARM
# g++ -m64 ./rwkv.cpp -I ./includg++ -m64 ./rwkv.cpp -I ./include/ -o ./build/rwkv -march=native -std=c++17 $vulk -ffast-math -O3 -pthreade/ -o ./build/rwkv -mcpu=native -std=c++17 -ffast-math -O3

# no avx512 support
# g++ ./rwkv.cpp -I ./include/ -o ./build/rwkv -g -march=haswell -O3 -std=c++17


# copy include/tokenizer/rwkv_vocab_v20230424.txt to ./build/
cp ./include/tokenizer/rwkv_vocab_v20230424.txt ./build/

# run ./build/rwkv
# cd ./build/
# ./rwkv