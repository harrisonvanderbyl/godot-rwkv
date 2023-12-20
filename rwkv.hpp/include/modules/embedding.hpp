#include "hvml/tensor.hpp"
class Embedding
{
    public:
        Tensor<float> weight;
        long max_batch;
        long max_seq;
        Tensor<float> buffer;
        Tensor<float> vkbuffer;
        Embedding(){
        }
        Embedding(Tensor<float> weighta, ulong max_batcha, ulong max_seqa){
            this->weight = weighta;
            this->max_batch = max_batcha;
            this->max_seq = max_seqa;
            this->buffer = Tensor<float>({max_batcha, max_seqa, weight.shape[1]});
        }
        Tensor<float> operator()(std::vector<std::vector<ulong>> indices){

            this->buffer.unsafereshape({indices.size(), indices[0].size(), this->weight.shape[1]});

            this->weight.gather(indices, this->buffer);
            
            if(this->vkbuffer.device.device_type.i == KHVMLVULKAN.i){
                this->vkbuffer.unsafereshape({indices.size(), indices[0].size(), this->weight.shape[1]});
                this->vkbuffer.loadVKBuffer(this->buffer);
                return this->vkbuffer;
            }
            
            return this->buffer;
        }

        void toVulkan(){
            this->vkbuffer = Tensor<float>({this->buffer.shape[0], this->buffer.shape[1], this->buffer.shape[2]},0.0f);
            this->vkbuffer.sendToVulkan();

        }
};