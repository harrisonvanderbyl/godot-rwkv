#ifndef LAYERNORM_HPP
#define LAYERNORM_HPP
#include "hvml/tensor.hpp"
class LayerNorm
{
    public:
        Tensor<float> weight;
        Tensor<float> bias;
        Tensor<float> buffer;
        
        LayerNorm(Tensor<float> weighti, Tensor<float> biasi, ulong max_batch, ulong max_seq){
            this->weight = weighti;
            this->bias = biasi;
            this->buffer = Tensor<float>({max_batch, max_seq, weight.shape[0]});


        }
        LayerNorm(){
        }
        template<typename T>
        Tensor<float,T> operator()(Tensor<float,T>& input){
            
            this->buffer.unsafereshape({input.shape[0], input.shape[1], input.shape[2]});

            input.layernorm(this->weight, this->bias, this->buffer);

            return this->buffer;
        }

        void toVulkan(){
            this->weight.sendToVulkan();
            this->bias.sendToVulkan();
            this->buffer.sendToVulkan();
        }

};

class GroupNorm
{
    public:
        Tensor<float> weight;
        Tensor<float> bias;
        Tensor<float> buffer;
        ulong head;
        GroupNorm(Tensor<float> weighti, Tensor<float> biasi, ulong headi, ulong max_batch, ulong max_seq){
            this->weight = weighti;
            this->bias = biasi;
            this->head = headi;
            this->bias.reshape({head, bias.shape[0]/head});
            this->weight.reshape({head, weight.shape[0]/head});
            ulong CC = this->weight.shape[1];
            // std::cout << "GroupNorm:" << this->weight.shape[0] << std::endl;
            
            this->buffer = Tensor<float>({max_batch, max_seq, head, CC },0.0);
            // std::cout << "GroupNorm:" << max_batch << std::endl;



        }
        GroupNorm(){
        }
        template<typename UT>
        Tensor<float,UT> operator()(Tensor<float,UT>& input){
           
            this->buffer.unsafereshape({input.shape[0], input.shape[1], this->head, this->weight.shape[1]});

            ulong B = input.shape[0];
            ulong T = input.shape[1];
            ulong C = input.shape[2];

            input.reshape({B, T, this->head, C/this->head});

            if (input.device.device_type.i == KHVMLCPU.i){
                 // iterate through B,T,H and layernorm each
                    for(ulong i = 0; i < B; i++){
                        for(ulong j = 0; j < T; j++){
                            for(ulong k = 0; k < this->head; k++){
                                // std::cout << "i:" << i << std::endl;
                                auto mbuftemp = this->buffer[i][j][k];
                                auto subweight = this->weight[k];
                                auto subbias = this->bias[k];
                                input[i][j][k].layernorm(subweight, subbias, mbuftemp);
                            }
                        }
                    }
            }
            else{
                
                auto B = input.data;
                auto A = this->weight.data;
                auto D = this->bias.data;
                auto C = this->buffer.data;

                auto Batch = input.shape[0];
                auto Seq = input.shape[1];
                auto Head = input.shape[2];
                auto Out = input.shape[3];

                // vuda
                auto stream_id = 0;
                auto kernalparams = vuda::dim3(Batch, Seq, Head);
                vuda::launchKernel("layernorm.spv", "main", stream_id, kernalparams, Batch, Seq, Head*Out ,Head, B, A, D, C);
                vuda::streamSynchronize(stream_id);
            }
           


            this->buffer.reshape({B, T, C});
            return this->buffer;
        }

        void toVulkan(){
            this->weight.sendToVulkan();
            this->bias.sendToVulkan();
            this->buffer.sendToVulkan();
        }

};


#endif