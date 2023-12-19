#ifndef HVMLGENERICADD_CPP
#define HVMLGENERICADD_CPP
#include "hvml/tensor.hpp"

template<>
void Tensor<float,HVMLCPU>::add(Tensor<float,HVMLCPU> &tensor, Tensor<float,HVMLCPU> &result)
{
    // #pragma omp parallel for
    for (uint64_t i = 0; i < this->data_size_in_elements; i += SIMD_WIDTH)
    {
        STORE(result.data + i, ADD(LOAD(this->data + i), LOAD(tensor.data + i)));
    }
}





#endif // HVMLGENERICADD_CPP