#ifndef LINEAR_HPP
#define LINEAR_HPP
#include "hvml/tensor.hpp"
#include "safetensors/safetensors.hpp"
#include <iostream>
class Linear
{
    public:
        Tensor<u_char> weight;
        Tensor<float> range;
        Tensor<float> offset;
        Tensor<float> buffer;
        bool quantized = false;
        bool isbf16 = false;
        bool onGPU = false;
        Tensor<float> vkbuffer;

        ulong batch_size = 0;
        ulong seq_len = 0;
        ulong hidden_size = 0;
        

        
        Linear(){
            
        }

        Linear(safetensors::safetensors_t& model, std::string prefix, ulong max_batch, ulong max_seq){
            if (model.contains(prefix + ".weight.zero")){
                this->range = model[prefix + ".weight.range"];
                this->offset = model[prefix + ".weight.zero"];
                this->quantized = true;
                auto temp = model.getUCHAR(prefix + ".weight");
                this->weight = temp;

            }else{

                const auto& meta = model.metas.at(prefix + ".weight");
            
                if (meta.dtype == TENSORTYPE::kBFLOAT_16){
                    // this->weight = model.getBF16(prefix + ".weight");
                    // this->isbf16 = true;
                }
                else{
                    // std::cout << "Exception:" << e.what() << std::endl;
                    // std::cout << "Linear:" << prefix << " is not BF16" << std::endl;
                    auto temp = model[prefix + ".weight"];
                    this->weight = *(Tensor<u_char>*)&temp;
                    this->isbf16 = false;
                    
                }
            }
            

            this->buffer = Tensor<float>({max_batch, max_seq, this->weight.shape[0]});

            this->batch_size = max_batch;
            this->seq_len = max_seq;
            this->hidden_size = this->weight.shape[0];
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
            this->vkbuffer = other.vkbuffer;
            this->batch_size = other.batch_size;
            this->seq_len = other.seq_len;
            this->hidden_size = other.hidden_size;

            
        }
        
    
        
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
                        this->vkbuffer.unsafereshape({input.shape[0], input.shape[1], this->weight.shape[0]});
                        this->buffer.unsafereshape({input.shape[0], input.shape[1], this->weight.shape[0]});
                        this->vkbuffer.sendToVulkan();

                        mzweight.matmul(mzrange, mzoffset, tempinput, *(Tensor<float,HVMLVULKAN>*)&this->vkbuffer);
                        
                        return *(Tensor<float,T>*)&this->vkbuffer;
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
                        0.0f);
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
            if (this->quantized){
                // std::cout << "toVulkan: " << this->weight << std::endl;
                this->weight.sendToVulkan();
                // std::cout << "toVulkan: " << this->range << std::endl;
                this->range.sendToVulkan();
                // std::cout << "toVulkan: " << this->offset << std::endl;
                this->offset.sendToVulkan();
            }
            else{
                std::cout << "toVulkan: " << this->weight << std::endl;
                ((Tensor<float>*)&this->weight)->sendToVulkan();
            }

            this->vkbuffer = Tensor<float>({this->batch_size, this->seq_len, this->hidden_size}, 0.0f);
            this->vkbuffer.sendToVulkan();

            this->onGPU = true;
        }

};

#endif