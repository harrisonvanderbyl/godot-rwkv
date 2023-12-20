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
        auto ss = this->data;
        auto out = y.data;

        uint B = r.shape[0];
        uint T = r.shape[1];
        uint C = r.shape[2];
        uint H = this->shape[1];

        // 1d
        uint bsize = H * T * (C / H);
      
        // 1d tensor
        uint tsize = H * (C / H);
        // 2d tensor
        uint ttsize = H * (C / H) * (C / H);

        // 1d
        uint hsize = (C / H);
        // 2d
        uint hhsize = (C / H) * (C / H);

        
        const int stream_id = 0;

        auto kernalparams = vuda::dim3(B, H, 1);
        vuda::launchKernel("./shaders/wkv5.glsl.spv", "main", stream_id, kernalparams, B, T, C, H, rr, kk, vv, ww, uu, this->data, out);
        
        
        vuda::streamSynchronize(stream_id);
        
        
    }

#endif // HVMLVULCANWKV5_CPP