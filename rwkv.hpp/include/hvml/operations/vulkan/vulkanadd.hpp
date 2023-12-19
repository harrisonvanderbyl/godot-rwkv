#ifndef HVMLVULCANADD_CPP
#define HVMLVULCANADD_CPP
#include "hvml/tensor.hpp"

template<>
void Tensor<float,HVMLVULKAN>::add(Tensor<float,HVMLVULKAN> &tensor, Tensor<float,HVMLVULKAN> &result)
    {
        float* aa = (float*)this->data;
        float* bb = (float*)tensor.data;
        
        auto stream_id = 0;
        auto threads = (this->data_size_in_elements)/(16*32);
        auto kernalparams = vuda::dim3(threads,1,1);
        vuda::launchKernel("./shaders/add.glsl.spv", "main", stream_id, kernalparams, this->data_size_in_elements, this->data, tensor.data, result.data);
        vuda::streamSynchronize(stream_id);
        
    }





#endif // HVMLVULCANADD_CPP