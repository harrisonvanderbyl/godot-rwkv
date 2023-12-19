# if --skip is passed, skip the build
if [ "$1" = "--skip" ]; then
    echo "Skipping build"
    cd ./build/
    # try to run ./build/vulkantest, and exit if it fails
    ./vulkantest || cd .. && exit 1

    cd ..
    exit 0
fi
source /opt/intel/oneapi/2024.0/oneapi-vars.sh 
icpx -m64 ./vulkantest.cpp -o  ./build/vulkantest -I./include -g -march=native -std=c++17 -O3 -lvulkan -ffast-math
cd ./build/
# try to run ./build/vulkantest, and exit if it fails
./vulkantest || cd .. && exit 1

cd ..