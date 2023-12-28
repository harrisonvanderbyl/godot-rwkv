#include <iostream>
#include <string>
#include <fstream>
#include "safetensors/safetensors.hpp"
#include "modules/embedding.hpp"
#include "modules/layernorm.hpp"
#include "modules/linear.hpp"
#include "modules/block.hpp"

class RWKV
{
    Embedding emb1;
    LayerNorm ln0;
    LayerNorm ln_out;
    Linear output;
    std::vector<Block> blocks;
    safetensors::safetensors_t model;

public:
    ulong layers;
    ulong max_batch_seq = 0;

    RWKV(std::string path, ulong max_batch = 1, ulong max_seq = 50)
    {
        max_batch_seq = max_batch * max_seq;
        std::ifstream inFile;
        inFile.open(path, std::ios::binary);
        model = safetensors::safetensors_t(inFile);
        
        // for (auto key : model.keys())
        // {
        //     std::cout << key << std::endl;
        // }


        auto keys = model.keys();
        layers = 0;
        for (auto key : keys)
        {
            if (std::string(key).find("blocks.") != std::string::npos)
            {
                if (std::string(key).find("att.time_mix_k") != std::string::npos)
                {
                    layers++;
                }
               
            }
        }

        // std::cout << "layers:" << layers << std::endl;

        auto t1o = model["emb.weight"];
        this->emb1 = Embedding(t1o, max_batch, max_seq);
        this->ln0 = LayerNorm(model["blocks.0.ln0.weight"], model["blocks.0.ln0.bias"], max_batch, max_seq);
        this->ln_out = LayerNorm(model["ln_out.weight"], model["ln_out.bias"], max_batch, max_seq);
        this->output = Linear(model, "head", max_batch, max_seq);
        for (size_t i = 0; i < layers; i++)
        {
            blocks.push_back(Block(model, i, max_batch, max_seq));
        }
    }

    Tensor<float> operator()(std::vector<std::vector<ulong>> input)
    {
        auto x = emb1(input);
        // std::cout << "x:" << x << std::endl;
        x = ln0(x);
        for (size_t i = 0; i < layers; i++)
        {
            x = blocks[i](x);
        }
        auto xm = ln_out(x);
        // std::cout << "xm:" << xm << std::endl;
        auto t3o = output(xm);
        // std::cout << "t3:" << t3 << std::endl;
        // std::cout << output.weight << std::endl;
        output.buffer.unsafereshape({x.shape[0], x.shape[1], t3o.shape[2]});
        if(t3o.device.device_type.i != KHVMLCPU.i){
            t3o.unloadVKBuffer(output.buffer);
            return output.buffer;
        }
        return output.buffer;
    }

    void get_state(std::map<std::string, Tensor<float>> state, size_t batchid = 0){
       
        
        for (size_t i = 0; i < layers; i++)
        {
            auto wkv = blocks[i].att.state[batchid];
            auto ts1 = blocks[i].att.timeshift.state[batchid];
            auto ts2 = blocks[i].ffn.timeshift.state[batchid];
            // std::cout << "wkv:" << wkv.shape[0] << " : " << wkv.shape[1] << std::endl;
            // std::cout << "ts1:" << ts1.shape[0] << " : " << ts1.shape[1] << std::endl;
            // std::cout << "ts2:" << ts2.shape[0] << " : " << ts2.shape[1] << std::endl;

            state["blocks." + std::to_string(i) + ".att.state"].clone(wkv);
            state["blocks." + std::to_string(i) + ".att.timeshift.state"].clone(ts1);
            state["blocks." + std::to_string(i) + ".ffn.timeshift.state"].clone(ts2);
            
        }
    }

    void set_state(std::map<std::string, Tensor<float>> state, size_t batchid = 0){
        for (size_t i = 0; i < layers; i++)
        {
            auto wkv = state["blocks." + std::to_string(i) + ".att.state"];
            auto ts1 = state["blocks." + std::to_string(i) + ".att.timeshift.state"];
            auto ts2 = state["blocks." + std::to_string(i) + ".ffn.timeshift.state"];

            // std::cout << "wkv:" << wkv.shape[0] << " : " << wkv.shape[1] << std::endl;
            // std::cout << "ts1:" << ts1.shape[0] << " : " << ts1.shape[1] << std::endl;
            // std::cout << "ts2:" << ts2.shape[0] << " : " << ts2.shape[1] << std::endl;
            blocks[i].att.state[batchid].clone(wkv);
            blocks[i].att.timeshift.state[batchid].clone(ts1);
            blocks[i].ffn.timeshift.state[batchid].clone(ts2);
            
        }
    }

    std::map<std::string, Tensor<float>> new_state(){
        std::map<std::string, Tensor<float>> state;
        for (size_t i = 0; i < layers; i++)
        {
            auto wkv = blocks[i].att.state[0];
            auto ts1 = blocks[i].att.timeshift.state[0];
            auto ts2 = blocks[i].ffn.timeshift.state[0];
            // std::cout << "wkv:" << wkv.shape[0] << " : " << wkv.shape[1] << std::endl;
            // std::cout << "ts1:" << ts1.shape[0] << " : " << ts1.shape[1] << std::endl;
            // std::cout << "ts2:" << ts2.shape[0] << " : " << ts2.shape[1] << std::endl;

            state["blocks." + std::to_string(i) + ".att.state"] = Tensor<float>(wkv.shape);
            state["blocks." + std::to_string(i) + ".att.state"].fill(0);
            state["blocks." + std::to_string(i) + ".att.timeshift.state"] = Tensor<float>(ts1.shape);
            state["blocks." + std::to_string(i) + ".att.timeshift.state"].fill(0);
            state["blocks." + std::to_string(i) + ".ffn.timeshift.state"] = Tensor<float>(ts2.shape);
            state["blocks." + std::to_string(i) + ".ffn.timeshift.state"].fill(0);
            
        }
        return state;
    }

    void toVulkan(int device = 0){
        cudaSetDevice(device);

        emb1.toVulkan();
        ln0.toVulkan();
        ln_out.toVulkan();
        output.toVulkan();

        for (size_t i = 0; i < layers; i++)
        {
            blocks[i].ln1.toVulkan();
            blocks[i].ln2.toVulkan();
            blocks[i].att.gate.toVulkan();
            blocks[i].att.key.toVulkan();
            blocks[i].att.value.toVulkan();
            blocks[i].att.receptance.toVulkan();
            blocks[i].att.output.toVulkan();
            blocks[i].att.ln_x.toVulkan();
            blocks[i].att.time_decay.sendToVulkan();
            blocks[i].att.time_faaaa.sendToVulkan();
            blocks[i].att.time_mix_g.sendToVulkan();
            blocks[i].att.time_mix_k.sendToVulkan();
            blocks[i].att.time_mix_r.sendToVulkan();
            blocks[i].att.time_mix_v.sendToVulkan();
            blocks[i].att.timeshift.toVulkan();
            blocks[i].att.state.sendToVulkan();
            blocks[i].att.buffer.sendToVulkan();

            blocks[i].ffn.key.toVulkan();
            blocks[i].ffn.value.toVulkan();
            blocks[i].ffn.receptance.toVulkan();
            blocks[i].ffn.time_mix_k.sendToVulkan();
            blocks[i].ffn.time_mix_r.sendToVulkan();
            blocks[i].ffn.timeshift.toVulkan();
            blocks[i].ffn.buffer.sendToVulkan();
  
        }
        
    }
};
