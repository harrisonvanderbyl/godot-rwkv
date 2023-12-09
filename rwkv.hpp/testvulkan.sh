g++ -m64 ./vulkantest.cpp -o ./build/vulkantest -I./include -g -march=native -std=c++17 -fopenmp  -flto  -fopenmp -funroll-loops -D_GLIBCXX_PARALLEL -lvulkan
cd ./build/
# try to run ./build/vulkantest, and exit if it fails
./vulkantest || cd .. && exit 1

cd ..