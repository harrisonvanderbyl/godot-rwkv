#ifndef HVMLVULCANWKV5_CPP
#define HVMLVULCANWKV5_CPP
#include "hvml/tensor.hpp"
template <>
void Tensor<float, HVMLVULKAN>::wkv5(Tensor<float,HVMLVULKAN> &r, Tensor<float,HVMLVULKAN> &k, Tensor<float,HVMLVULKAN> &v, Tensor<float,HVMLVULKAN> &w, Tensor<float,HVMLVULKAN> &u, Tensor<float,HVMLVULKAN> &y)
    {

        auto rr = r.data;
        auto kk = k.data;
        auto vv = v.data;
        auto ww = w.data;
        auto uu = u.data;
        auto out = y.data;

        uint32_t B = r.shape[0];
        uint32_t T = r.shape[1];
        uint32_t C = r.shape[2];
        uint32_t H = this->shape[1];


        
        const int stream_id = 0;

        auto kernalparams = vuda::dim3(B, H, 1);
        vuda::launchKernel("./shaders/wkv5.glsl.spv", "main", stream_id, kernalparams, B, T, C, H, rr, kk, vv, ww, uu, this->data, out);
        
        
        vuda::streamSynchronize(stream_id);
        
        
    }

#endif // HVMLVULCANWKV5_CPP