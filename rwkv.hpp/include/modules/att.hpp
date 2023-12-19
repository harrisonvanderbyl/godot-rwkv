#include "hvml/tensor.hpp"

#include "safetensors/safetensors.hpp"
#include "modules/timeshift.hpp"
#include "modules/linear.hpp"
class RWKV_5_ATT
{
    public:
        uint head_size = 64;
        uint n_head; 
        TimeShift timeshift;
        Tensor<float> time_mix_k;
        Tensor<float> time_mix_v;
        Tensor<float> time_mix_r;
        Tensor<float> time_mix_g;
        Tensor<float> time_decay;
        Tensor<float> time_faaaa;
        Tensor<float> state;
        Tensor<float> buffer;
        Tensor<float> buffer1;
        Tensor<float> buffer2;
        Tensor<float> buffer3;
        Linear receptance;
        Linear key;
        Linear value;
        Linear gate;
        Linear output;
        GroupNorm ln_x;
        int layer = 0;

        RWKV_5_ATT(){
        }
        
        RWKV_5_ATT(int layerID, safetensors::safetensors_t& model, ulong max_batch, ulong max_seq){
            // std::cout << "RWKV_5_ATTcreate:" << layerID << std::endl;
            std::string prefix = "blocks." + std::to_string(layerID) + ".att.";
            this->layer = layerID;
            this->time_mix_k = model[prefix + "time_mix_k"][0][0];
            this->time_mix_v = model[prefix + "time_mix_v"][0][0];
            this->time_mix_r = model[prefix + "time_mix_r"][0][0];
            this->time_mix_g = model[prefix + "time_mix_g"][0][0];

            auto dims = this->time_mix_k.shape[0];

            // std::cout << "time_mix_k:" << dims << std::endl;

            this->n_head = dims/this->head_size;
            this->state = Tensor<float>({max_batch*max_seq, this->n_head , this->head_size, this->head_size},0.0f);
            // std::cout << "n_head:" << this->n_head << std::endl;
            
            this->time_decay = model[prefix + "time_decay"];
            this->time_faaaa = model[prefix + "time_faaaa"];
            this->buffer = Tensor<float>({max_batch, max_seq, dims},0.0);
            this->buffer1 = Tensor<float>({max_batch, max_seq, dims},0.0);
            this->buffer2 = Tensor<float>({max_batch, max_seq, dims},0.0);
            this->buffer3 = Tensor<float>({max_batch, max_seq, dims},0.0);

            this->timeshift = TimeShift(max_batch, max_seq, dims);

            this->receptance = Linear(model, prefix + "receptance", max_batch, max_seq);
            this->key = Linear(model, prefix + "key", max_batch, max_seq);
            this->value = Linear(model, prefix + "value", max_batch, max_seq);
            this->gate = Linear(model, prefix + "gate", max_batch, max_seq);
            this->output = Linear(model, prefix + "output", max_batch, max_seq);
            this->ln_x = GroupNorm(model[prefix + "ln_x.weight"], model[prefix + "ln_x.bias"], n_head, max_batch, max_seq);
            
        }



        Tensor<float> operator()(Tensor<float>& input){
            // std::cout << "RWKV_5_ATT:" << this->layer << std::endl;
            this->buffer.unsafereshape({input.shape[0], input.shape[1], input.shape[2]});
         
            auto xx = this->timeshift(input);

            this->buffer1.unsafereshape({input.shape[0], input.shape[1], input.shape[2]});
            this->buffer2.unsafereshape({input.shape[0], input.shape[1], input.shape[2]});
            this->buffer3.unsafereshape({input.shape[0], input.shape[1], input.shape[2]});
            this->buffer.unsafereshape({input.shape[0], input.shape[1], input.shape[2]});

           
            this->time_mix_k.lerp(xx, input, this->buffer);
            auto k = this->key.buffer;
            this->time_mix_v.lerp(xx, input, this->buffer1);
            auto v = this->value.buffer;
            this->time_mix_r.lerp(xx, input, this->buffer2);
            auto r = this->receptance.buffer;
            this->time_mix_g.lerp(xx, input, this->buffer3);
            auto gv = this->gate.buffer;  

            k.unsafereshape({input.shape[0], input.shape[1], input.shape[2]});
            v.unsafereshape({input.shape[0], input.shape[1], input.shape[2]});
            r.unsafereshape({input.shape[0], input.shape[1], input.shape[2]});
            gv.unsafereshape({input.shape[0], input.shape[1], input.shape[2]});

            matmul(
                this->key.weight,this->key.range,this->key.offset,this->buffer,k,
                this->value.weight,this->value.range,this->value.offset,this->buffer1,v,
                this->receptance.weight,this->receptance.range,this->receptance.offset,this->buffer2,r,
                this->gate.weight,this->gate.range,this->gate.offset,this->buffer3,gv
            );


    
            this->state.wkv5(r,k,v,this->time_decay,this->time_faaaa, this->buffer);
          
            auto xxa = this->ln_x(this->buffer);

            gv.swishmult(xxa,this->buffer);
               
            auto xout = this->output(this->buffer);

             
            return xout;


        }

};