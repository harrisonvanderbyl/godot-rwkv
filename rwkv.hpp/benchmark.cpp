#include <iostream>
#include <string>
#include <fstream>
#include "rwkv.hpp"
#include "sampler/sample.hpp"
int main(){
    std::cout << "Hello World" << std::endl;
    std::string path = "./model.safetensors";


    RWKV model(path, 1024, 2);

    std::cout << "Model loaded" << std::endl;

    ulong tokenstogen = 100;

    for (int i = 0; i < 100; i++)
    {
        std::cout << "Generating token " << i << std::endl;
        std::vector<std::vector<ulong>> input = {};
        for (int j = 0; j < 1024; j++)
        {
            input.push_back({0});
        }

        auto time = std::chrono::high_resolution_clock::now();

        auto logits = model(input);

        auto time2 = std::chrono::high_resolution_clock::now();

        std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time).count() << "ms" << std::endl;
        
    }


   

    
    return 0;
}