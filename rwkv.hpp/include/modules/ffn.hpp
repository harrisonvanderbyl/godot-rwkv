#include "hvml/tensor.hpp"

#include "safetensors/safetensors.hpp"
#include "modules/timeshift.hpp"
#include "modules/linear.hpp"
class FFN
{
    public:
        uint head_size = 64;
        uint n_head; 
        TimeShift timeshift;
        Tensor<float> time_mix_k;
        Tensor<float> time_mix_r;
        Linear receptance;
        Linear key;
        Linear value;
        Tensor<float> buffer;

        FFN(){
        }
        
        FFN(int layerID, safetensors::safetensors_t& model, ulong max_batch, ulong max_seq){
            std::string prefix = "blocks." + std::to_string(layerID) + ".ffn.";

            this->time_mix_k = model[prefix + "time_mix_k"][0][0];
            this->time_mix_r = model[prefix + "time_mix_r"][0][0];

            auto dims = this->time_mix_k.shape[0];
            // std::cout << "dims:" << dims << std::endl;

            this->timeshift = TimeShift(max_batch, max_seq, dims);

            this->receptance = Linear(model, prefix + "receptance", max_batch, max_seq);
            this->key = Linear(model, prefix + "key", max_batch, max_seq);
            this->value = Linear(model, prefix + "value", max_batch, max_seq);
            this->buffer = Tensor<float>({max_batch, max_seq, this->key.weight.shape[0]});
        }
        Tensor<float> operator()(Tensor<float>& input){

            
            // std::cout << "xx:" << input << std::endl;
         
            // auto cbuff = Tensor<float>(input.shape, this->buffer.data);
            this->buffer.unsafereshape({input.shape[0], input.shape[1], input.shape[2]});
            // std::cout << "dims:" << input.shape[2] << std::endl;
            // std::cout << "FFNInput" << input[0][0] << std::endl;
            auto xx = this->timeshift(input);

            this->time_mix_k.lerp(xx, input, this->buffer);
            auto k = this->key(this->buffer);

           
            this->time_mix_r.lerp(xx, input, this->buffer);
            auto r = this->receptance(this->buffer);

            this->buffer.unsafereshape({k.shape[0], k.shape[1], k.shape[2]});

            k.relusquare(this->buffer);

            auto v = this->value(this->buffer); 

            r.sigmoidmult(v,this->buffer);
            
            return  this->buffer;


        }

};