#include "hvml/tensor.hpp"
#include "modules/layernorm.hpp"
#include "safetensors/safetensors.hpp"
#include "modules/att.hpp"
#include "modules/ffn.hpp"
class Block
{
    public:
        LayerNorm ln1;
        LayerNorm ln2;
        RWKV_5_ATT att;
        FFN ffn;
        int layerid = -1;
        
        Block(safetensors::safetensors_t& model, int layerID, ulong max_batch, ulong max_seq){
            this->layerid = layerID;
            // std::cout << "Blockcreate:" << layerID << std::endl;
            this->ln1 = LayerNorm(model["blocks." + std::to_string(layerID) + ".ln1.weight"], model["blocks." + std::to_string(layerID) + ".ln1.bias"], max_batch, max_seq);
            this->ln2 = LayerNorm(model["blocks." + std::to_string(layerID) + ".ln2.weight"], model["blocks." + std::to_string(layerID) + ".ln2.bias"], max_batch, max_seq);
            // std::cout << "Blockcreate1:" << layerID << std::endl;
            this->att = RWKV_5_ATT(layerID, model, max_batch, max_seq);
            // std::cout << "Blockcreate2:" << layerID << std::endl;
            this->ffn = FFN(layerID, model, max_batch, max_seq);
            // std::cout << "Blockcreate3:" << layerID << std::endl;
        }
        Tensor<float> operator()(Tensor<float> input){

            auto l1 = this->ln1(input);
            
            // std::cout << "l1:" << l1 << std::endl;
            
            this->att(l1).add(input, l1);
            
            auto l2 = this->ln2(l1);
            // std::cout << "l3:" << l2[0][0] << std::endl;
            auto temp = this->ffn(l2);
            // std::cout << "FFNOUT:" << temp[0][0] << std::endl;
            temp.add(l1, l2);
            // std::cout << "l4:" << l2[0][0] << std::endl;
            // while(1){

            // }
            return l2;
            
        }

};