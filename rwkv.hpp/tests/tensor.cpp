#include "../include/hvml/tensor.hpp"
#include "../include/modules/timeshift.hpp"

#include <iostream>

// assert
#include <cassert>

int main()
{
    std::cout << "Tensor Test" << std::endl;
    Tensor<float> tensor({16, 3, 4});
    assert(tensor.data_size_in_elements == 16*3*4);
    assert(tensor.data_size_in_bytes == 16*3*4*sizeof(float));
    assert(tensor.shape[0] == 16);
    assert(tensor.shape[1] == 3);
    assert(tensor.shape[2] == 4);

    std::cout << "Allocation test passed\n" << std::endl;

    tensor.fill(12.0f);
    std::cout << "Filling test passed\n" << std::endl;

    assert(tensor.data_size_in_elements == 16*3*4);
    for (int i = 0; i < tensor.data_size_in_elements; i++)
    {
        assert(tensor.data[i] == 12.0f);
    }

    std::cout << "Allocation and filling test passed\n" << std::endl;
    
    Tensor<float> tensor2({16, 3, 4}, 0.9);
    assert(tensor2.data_size_in_elements == 16*3*4);
    assert(tensor2.data_size_in_bytes == 16*3*4*sizeof(float));
    assert(tensor2.shape[0] == 16);
    assert(tensor2.shape[1] == 3);
    assert(tensor2.shape[2] == 4);

    std::cout << "Allocation test2 passed\n" << std::endl;

    assert(tensor2.data_size_in_elements == 16*3*4);
    for (int i = 0; i < tensor2.data_size_in_elements; i++)
    {
        assert(tensor2.data[i] == 0.9f);
    }

    std::cout << "Allocation and filling test2 passed\n" << std::endl;

    Tensor<float> tensor3({16, 3, 4});

    tensor.add(tensor2, tensor3);

    for (int i = 0; i < tensor3.data_size_in_elements; i++)
    {
        assert(tensor3.data[i] == 12.9f);
    }

    std::cout << "Addition test passed\n" << std::endl;

// allocate with 64 aligned memory
    float* array = (float*)aligned_alloc(64, 3*16*sizeof(float));
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            array[i*16+j] = float(i*100 + j);
        }
    }

    Tensor<float> tensor4({3,16}, array);

    for (int i = 0; i < tensor4.data_size_in_elements; i++)
    {
        assert(tensor4.data[i] == array[i]);
    }

    std::cout << "Preallocated Tensor Data Test Passed\n" << std::endl;


    Tensor<float> tensor5({1, 4, 16});
    std::cout << "Tensor5 loaded " << std::endl;

    tensor5.fill(0.0f);
    std::cout << "Tensor5 filled " << std::endl;
    tensor4.gather({{0,1,0,2}},tensor5);
    std::cout << "Tensor4 gathered " << std::endl;

    std::cout << tensor5[0][3] << std::endl;

    // tensor5[0][0] == 1.0f; [0][1] == 100.0f; [0][2] == 1.0f; [0][3] == 200.0f;
    assert(tensor5[0][0][1] == 1.0f);
    assert(tensor5[0][1][1] == 101.0f);
    assert(tensor5[0][2][1] == 1.0f);
    assert(tensor5[0][3][1] == 201.0f); 

    std::cout << "Gather Test Passed\n" << std::endl;

    Tensor<float> tensor6({16, 16});
    tensor6.fill(0.5f);

    Tensor<float> tensor7({4, 21, 16});
    tensor7.fill(0.25f);

    Tensor<float> tensor8({4, 21, 16});

    tensor6.matmul(tensor7, tensor8);
    std::cout << tensor8[3][15] << std::endl;
    std::cout << "Matmul Test Passed\n" << std::endl;

    Tensor<float> tensor9 = Tensor<float>({16, 16});

    std::cout << "Tensor9 loaded " << std::endl;

    tensor9.fill(0.5f);

    std::cout << "Tensor9: " << tensor9[0] << std::endl;

    Tensor<float> tensor10({16, 16});
    tensor10.fill(1.0f);

    std::cout << "Tensor10: " << tensor10[0] << std::endl;


    Tensor<float> out({16, 16});

    Tensor<float> tensor11({16}); 
    tensor11.fill(2.0f);
    
    tensor11.lerp(tensor9, tensor10, out);

    std::cout << out[0] << std::endl;

    std::cout << "Lerp Test Passed\n" << std::endl;
    // std::cout << "Tensor 5 shape: " << tensor5.shape[0] << ", " << tensor5.shape[1] << ", " << tensor5.shape[2] << std::endl;
    // tensor4.gather({{0,1,0,2}},tensor5);

    // LayerNorm test
    /*
    see pythoneq.py
    tensor([[ 0.7190,  0.9998,  0.9528, -0.4320,  0.7933, -0.1446, -0.2752,  0.6127,
          0.5435,  0.9094,  1.0250,  0.3071, -0.5466,  0.3347,  0.9648,  2.0692],
        [ 1.5440,  1.0835,  0.8994,  1.4524,  0.0171, -0.4639,  0.9227,  0.6095,
          1.2399,  0.6054,  0.7828,  0.9634, -0.2421,  0.3577, -0.7893,  0.1603]])
    */
    float aaa[32] = {0.4539, 0.6607, 0.7687, 0.0256, 0.6968, 0.0456, 0.0430, 0.9797, 0.0495,
         0.9150, 0.7107, 0.2377, 0.1771, 0.0177, 0.5180, 0.9685,
        0.8520, 0.8278, 0.7646, 0.7894, 0.2996, 0.0799, 0.6645, 0.9915, 0.8439,
         0.6807, 0.6676, 0.6729, 0.4684, 0.2487, 0.0114, 0.2972};

    Tensor<float> tensor12({2, 16}, aaa);

    float bbb[16] = {0.8531, 0.2775, 0.2419, 0.9635, 0.4775, 0.5776, 0.8121, 0.1070, 0.3353,
            0.3296, 0.6165, 0.6863, 0.7249, 0.2105, 0.8279, 0.7959};

    Tensor<float> tensor13({16}, bbb);

    float ccc[16] = {0.7199, 0.8386, 0.7388, 0.7301, 0.4675, 0.5195, 0.6645, 0.4545, 0.9254,
            0.4821, 0.5802, 0.7253, 0.0187, 0.5933, 0.8163, 0.9176};
    
    Tensor<float> tensor14({16}, ccc);

    Tensor<float> tensor15({2, 16});

    tensor12.layernorm(tensor13, tensor14, tensor15);

    assert(abs(tensor15[1][13] - 0.357714) < 0.0001);
    
    std::cout << "LayerNorm Test Passed\n" << std::endl;

    auto timeShift = TimeShift(2, 4, 16);

    auto input = Tensor<float>({2, 3, 16});
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 16; k++)
            {
                *input[i][j][k].data = float(i*1000 + j*100 + k);
            }
        }
    }

    std::cout << "Input: " << input[0][1] << std::endl;

    auto output = timeShift(input);

    std::cout << "Output: " << output[0][1] << std::endl;
    std::cout << "state: " << timeShift.state[0][0] << std::endl;


    
}