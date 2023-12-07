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
        Tensor<float> operator()(Tensor<float>& input){
            auto mbuff = Tensor<float>({input.shape[0], input.shape[1], this->weight.shape[0]},
                this->buffer.data);

            input.layernorm(this->weight, this->bias, mbuff);

            return mbuff;
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
        Tensor<float> operator()(Tensor<float>& input){
            auto mbuff = Tensor<float>({input.shape[0], input.shape[1], this->head, this->weight.shape[1]},
                this->buffer.data);

            ulong B = input.shape[0];
            ulong T = input.shape[1];
            ulong C = input.shape[2];

            
            // std::cout << "input:" << input.shape[0] << std::endl;
            // std::cout << "input:" << input.shape[1] << std::endl;
            // std::cout << "input:" << input.shape[2] << std::endl;

            input.reshape({B, T, this->head, C/this->head});


            // std::cout << "weight:" << this->weight.shape[0] << std::endl;

            // std::cout << "bias:" << this->bias.shape[0] << std::endl;

            // iterate through B,T,H and layernorm each
            for(ulong i = 0; i < B; i++){
                for(ulong j = 0; j < T; j++){
                    for(ulong k = 0; k < this->head; k++){
                        // std::cout << "i:" << i << std::endl;
                        auto mbuftemp = mbuff[i][j][k];
                        auto subweight = this->weight[k];
                        auto subbias = this->bias[k];
                        input[i][j][k].layernorm(subweight, subbias, mbuftemp);
                    }
                }
            }
            mbuff.reshape({B, T, C});
            return mbuff;
        }

};

// class GroupNorm
// {
//     public:
//         Tensor<float> weight;
//         Tensor<float> bias;
//         uint group;
//         GroupNorm(Tensor<float> weight, Tensor<float> bias, uint group){
//             this->weight = weight;
//             this->bias = bias;
//             this->group = group;
//             }
//         GroupNorm(){
//             this->weight = tensor::Tensor();
//             this->bias = tensor::Tensor();
//             this->group = 1;
//         }
//         Tensor<float> operator()(tensor::Tensor& input){
//             uint64_t inpsize = 1;
//             for (int i = 0; i < input.shape.size(); i++){
//                 inpsize *= input.shape[i];
//             }
//             input.reshape({this->group,inpsize/this->group});
        
//             auto inc = input;
//             for (int i = 0; i < inc.shape[0]; i++){
//                 auto sub = inc[i];
//                 auto mean = sub.mean();
//                 auto std = sub.std();
//                 auto nsub = (sub - mean) / sqrt(std);
//                 for (int j = 0; j < sub.shape[0]; j++){
//                     ((float*)(sub.data.first)) [j] = ((float*)(nsub.data.first)) [j];
//                 }
//                 // sub = ((sub - mean) / sqrt(std)).data;
//             }
//             inc.reshape({inpsize});
//             auto nout = inc.fma(this->weight , this->bias);
//             return nout;
            
//         }

// };


#endif