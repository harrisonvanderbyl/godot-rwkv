#ifndef TIMESHIFT_HPP
#define TIMESHIFT_HPP
#include "hvml/tensor.hpp"
#include "safetensors/safetensors.hpp"
class TimeShift
{
    public:
        uint shiftamount = 1;

        Tensor<float> state;
        
        Tensor<float> buffer;
        ulong max_batch;
        ulong max_seq;
        ulong dims;
        
        TimeShift(){
        }

        TimeShift(const ulong max_batchi, const ulong max_seqi, const ulong dimsi){
            std::vector<ulong> size = {max_batchi, max_seqi, dimsi};
            this->buffer = Tensor<float>(size,0.0f);
            std::vector<ulong> state_size = {max_batchi, 1UL, dimsi};
            // std::cout << "TimeShift:" << state_size[0] << std::endl;
            this->state = Tensor<float>(state_size,0.0f);
            
            this->max_batch = max_batchi;
            this->max_seq = max_seqi;
            this->dims = dimsi;
            
        }

        Tensor<float> operator()(Tensor<float> input){
            this->buffer.unsafereshape({input.shape[0], input.shape[1], input.shape[2]});
            auto batches = input.shape[0];
            auto seq = input.shape[1];

            if (this->buffer.device.device_type.i == KHVMLCPU.i){
                for (size_t i = 0; i < batches; i++){
                    this->buffer[i][0].clone(this->state[i][0]);
                    for (size_t j = 0; j < seq; j++){
                        if (j > 0){
                            this->buffer[i][j].clone(input[i][j-1]);
                        }
                        else{
                            this->state[i][0].clone(input[i][seq-1]);
                        }
                    }
                }
            }
            else{

                auto B = input.data;
                auto C = this->buffer.data;
                auto A = this->state.data;

                auto Batch = input.shape[0];
                auto Seq = input.shape[1];
                auto Out = input.shape[2];

                // vuda
                auto stream_id = 0;
                const int CHUNKSIZE = 128;
                auto kernalparams = vuda::dim3(Batch, 1, 1);
                vuda::launchKernel("./shaders/timeshift.glsl.spv", "main", stream_id, kernalparams, Batch, Seq, Out,1, A, B, C);
                vuda::streamSynchronize(stream_id);
                
            }


            return this->buffer;            
        }


        void toVulkan(){
            this->state.sendToVulkan();
            this->buffer.sendToVulkan();
        }

};

#endif