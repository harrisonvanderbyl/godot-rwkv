#include "hvml/tensor.hpp"
class Embedding
{
    public:
        Tensor<float> weight;
        long max_batch;
        long max_seq;
        Tensor<float> buffer;
        Embedding(){
        }
        Embedding(Tensor<float> weight, ulong max_batch, ulong max_seq){
            this->weight = weight;
            this->max_batch = max_batch;
            this->max_seq = max_seq;
            this->buffer = Tensor<float>({max_batch, max_seq, weight.shape[1]});
        }
        Tensor<float> operator()(std::vector<std::vector<ulong>> indices){

            this->buffer.unsafereshape({indices.size(), indices[0].size(), this->weight.shape[1]});

            this->weight.gather(indices, this->buffer);
            
            
            return this->buffer;
        }

        void toVulkan(){
            this->weight.sendToVulkan();
            this->buffer.sendToVulkan();
        }
};