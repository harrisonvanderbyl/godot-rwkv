#ifndef LINEAR_HPP
#define LINEAR_HPP
#include "hvml/tensor.hpp"
#include "safetensors/safetensors.hpp"
#include <iostream>
class Linear
{
    public:
        Tensor<bfloat16> weight;
        Tensor<float> range;
        Tensor<float> offset;
        Tensor<float> buffer;
        bool quantized = false;
        bool isbf16 = false;
        bool onGPU = false;
        Tensor<float> vkbuffer;
        

        
        Linear(){
            
        }

        Linear(safetensors::safetensors_t& model, std::string prefix, ulong max_batch, ulong max_seq){
            if (model.contains(prefix + ".weight.zero")){
                this->range = model[prefix + ".weight.range"];
                this->offset = model[prefix + ".weight.zero"];
                this->quantized = true;
                auto temp = model.getUCHAR(prefix + ".weight");
                this->weight = *(Tensor<bfloat16>*)&temp;

            }else{

                const auto& meta = model.metas.at(prefix + ".weight");
            
                if (meta.dtype == TENSORTYPE::kBFLOAT_16){
                    this->weight = model.getBF16(prefix + ".weight");
                    this->isbf16 = true;
                }
                else{
                    // std::cout << "Exception:" << e.what() << std::endl;
                    // std::cout << "Linear:" << prefix << " is not BF16" << std::endl;
                    auto temp = model[prefix + ".weight"];
                    this->weight = *(Tensor<bfloat16>*)&temp;
                    this->isbf16 = false;
                    
                }
            }
            

            this->buffer = Tensor<float>({max_batch, max_seq, this->weight.shape[0]});
        }

        // Copy constructor
        Linear(const Linear& other){
            this->weight = other.weight;
            this->range = other.range;
            this->offset = other.offset;
            this->quantized = other.quantized;
            this->buffer = other.buffer;
            this->isbf16 = other.isbf16;
            this->onGPU = other.onGPU;

            
        }
        
        // Tensor<float,HVMLVULKAN> operator()(Tensor<float, HVMLVULKAN>& input) {
        //     if (this->quantized){
        //         auto devicemap = input.device.device_type.i + this->weight.device.device_type.i + this->range.device.device_type.i + this->offset.device.device_type.i;

        //         assert (devicemap == 4);

        //         auto mzweight = this->weight.sendToVulkan<uint8_t>();
        //         auto mzrange = this->range.sendToVulkan();
        //         auto mzoffset = this->offset.sendToVulkan();
        //         auto tempinput = input.sendToVulkan();

        //         mzweight.matmul(mzrange, mzoffset, tempinput, this->vkbuffer);
                
        //         return vkbuffer;

        //     }
        // }
        
        template<typename T>
        Tensor<float,T> operator()(Tensor<float,T>& input) {
               if (this->quantized){

                    if(this->onGPU){
                        auto devicemap = input.device.device_type.i + this->weight.device.device_type.i + this->range.device.device_type.i + this->offset.device.device_type.i;

                        assert (devicemap == 4);

                        auto mzweight = this->weight.sendToVulkan<uint8_t>();
                        auto mzrange = this->range.sendToVulkan();
                        auto mzoffset = this->offset.sendToVulkan();
                        auto tempinput = input.sendToVulkan();
                        auto buff = this->vkbuffer.sendToVulkan();

                        mzweight.matmul(mzrange, mzoffset, tempinput, buff);
                        
                        return vkbuffer;
                    }
                    else{
                        auto mbuff = Tensor<float>({input.shape[0], input.shape[1], this->weight.shape[0]},
                            this->buffer.data);

                        ((Tensor<uint8_t>*)(&this->weight))->matmul(this->range, this->offset, input, mbuff);

                        return mbuff;
                    }

                }
                else{
                    auto mbuff = Tensor<float>({input.shape[0], input.shape[1], this->weight.shape[0]},
                        0.0);
                    if(this->isbf16){
                        // ((Tensor<bfloat16>*)(&this->weight))->matmul(input, mbuff, true);
                    }
                    else{
                        if (this->onGPU){
                            // mbuff.sendToVulkan();
                            // auto tempinput = input.sendToVulkan();
                            // (*(Tensor<float,VKTensorInfo<float>>*)&this->weight).matmul(tempinput, *(Tensor<float,VKTensorInfo<float>>*)&mbuff);
                            // ;
                            // input.receiveFromVulkan();
                            // (*(Tensor<float,VKTensorInfo<float>>*)&mbuff).receiveFromVulkan();
                            // // std::cout << "mbuff:" << mbuff << std::endl;
                            // return mbuff;
                        }
                        else{                        
                            // ((Tensor<float>*)(&this->weight))->matmul(input, mbuff);
                        }
                    }
                    return mbuff;
                }     
        }

        void toVulkan(){
            cudaSetDevice(0);
            if (this->quantized){
                // std::cout << "toVulkan: " << this->weight << std::endl;
                this->weight.sendToVulkan<uint8_t>();
                // std::cout << "toVulkan: " << this->range << std::endl;
                this->range.sendToVulkan();
                // std::cout << "toVulkan: " << this->offset << std::endl;
                this->offset.sendToVulkan();
            }
            else{
                std::cout << "toVulkan: " << this->weight << std::endl;
                ((Tensor<float>*)&this->weight)->sendToVulkan();
            }

            this->vkbuffer = Tensor<float>({this->buffer.shape[0], this->buffer.shape[1], this->buffer.shape[2]}, 0.0);
            this->vkbuffer.sendToVulkan();

            this->onGPU = true;
        }

};

#endif