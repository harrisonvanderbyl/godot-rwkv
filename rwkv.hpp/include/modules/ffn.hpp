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
            /*
            FFNInput tensor([ 0.2754, -0.0405, -0.2988,  0.1033,  0.2481])
            FFNInput2 tensor([-0.1140,  0.0138,  0.1255, -0.0486, -0.0703])
            FFNInput3 tensor([ 0.1592,  0.0018,  0.0423, -0.0434, -0.0342])
            FFNKey tensor([-0.4409, -1.2823, -1.1009, -1.1669, -0.9889])
            FFNKey2 tensor([0., 0., 0., 0., 0.])
            FFNValue tensor([ 6.7170, -0.0981, -1.7468,  1.5263,  5.9379])
            */
            // auto cbuff = Tensor<float>(input.shape, this->buffer.data);
            this->buffer.unsafereshape({input.shape[0], input.shape[1], input.shape[2]});
            // std::cout << "dims:" << input.shape[2] << std::endl;
            // std::cout << "FFNInput" << input[0][0] << std::endl;
            auto xx = this->timeshift(input);
            this->time_mix_k.lerp(xx, input, this->buffer);
            // std::cout << "FFNInput2" << cbuff[0][0] << std::endl;
            auto k = this->key(this->buffer);
            

           
            this->time_mix_r.lerp(xx, input, this->buffer);
            // std::cout << "FFNInput3" << cbuff[0][0] << std::endl;
            auto r = this->receptance(this->buffer);
            // std::cout << "FFNKey" << k[0][0] << std::endl;
            // std::cout << "kshape:" << k.shape[0] << ":" << k.shape[1] << ":" << k.shape[2] << std::endl;
            
            k.relu(this->buffer);
            this->buffer.multiply(this->buffer,this->buffer);
            // std::cout << "FFNKey2" << cbuff[0][0] << std::endl;
            // std::cout << "cbufshape:" << cbuff.shape[0] << ":" << cbuff.shape[1] << ":" << cbuff.shape[2] << std::endl;
            auto v = this->value(this->buffer); 
            // std::cout << "vshape:" << v.shape[0] << ":" << v.shape[1] << ":" << v.shape[2] << std::endl;
            /*
            FFNValue tensor([ 6.7170, -0.0981, -1.7468,  1.5263,  5.9379])
            */
            // std::cout << "FFNValue" << v[0][0] << std::endl;
         
            r.sigmoid(this->buffer);

            // std::cout << "r.sigmoid" << cbuff[0][0] << std::endl;

            v.multiply(this->buffer, this->buffer);

            // std::cout << "v.multiply" << cbuff[0][0] << std::endl;

            // while(1){

            // }
            
            return  this->buffer;


        }

};