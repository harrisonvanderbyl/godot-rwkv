g++ ./tensor.cpp -g -march=native -O3 -std=c++17 -fopenmp  -flto  -fopenmp -funroll-loops -D_GLIBCXX_PARALLEL -I../include
./a.out
# g++ ./tensor.cpp -g -O3 -std=c++17 -fopenmp  -flto  -fopenmp -funroll-loops -D_GLIBCXX_PARALLEL
# ./a.out
# g++ ./tensor.cpp -g -march=native -O3 -std=c++17 -fopenmp  -flto  -fopenmp -funroll-loops -D_GLIBCXX_PARALLEL
# ./a.out