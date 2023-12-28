//
// Created by mfuntowicz on 3/28/23.
//

#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <span>

#include "hvml/tensor.hpp"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace safetensors {
    

    
    struct metadata_t {
        TENSORTYPE dtype;
        std::vector<size_t> shape;
        std::pair<size_t, size_t> data_offsets;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(metadata_t, dtype, shape, data_offsets)

    /**
     *
     */
    class safetensors_t {

    public:

        std::unordered_map<std::string, const metadata_t> metas;
        
       
        const char* storage = nullptr;
        /**
         *
         * @return
         */
        inline size_t size() const { return metas.size(); }

        /**
         *
         * @param name
         * @return
         */
        
         Tensor<float> operator[](const char* name) const;
         Tensor<float> operator[](std::string name) const{
                return operator[](name.c_str());
         }

        Tensor<bfloat16> getBF16(const char* name);
        Tensor<bfloat16> getBF16(std::string name){
            return getBF16(name.c_str());
        }
        
        Tensor<u_char> getUCHAR(const char* name);
        Tensor<u_char> getUCHAR(std::string name){
            return getUCHAR(name.c_str());
        }
         

         /**
         *
         * @param name
         * @return
         */
        inline std::vector<const char*> keys() const {
            std::vector<const char*> keys;
            keys.reserve(metas.size());
            for (auto &item: metas) {
                keys.push_back(item.first.c_str());
            }
            return keys;
        }

        // contains key
        inline bool contains(const char* name) const {
            auto keys = this->keys();
            bool found = false;

            for (auto key : keys){
                if (strcmp(key, name) == 0){
                    found = true;
                }

            }
            return found;
        }
        inline bool contains(std::string name) const {
            return contains(name.c_str());
        }

        safetensors_t(){};

        safetensors_t(std::basic_istream<char> &in) {
                uint64_t header_size = 0;

                // todo: handle exception
                in.read(reinterpret_cast<char *>(&header_size), sizeof header_size);

                std::vector<char> meta_block(header_size);
                in.read(meta_block.data(), static_cast<std::streamsize>(header_size));
                const auto metadatas = json::parse(meta_block);

                // How many bytes remaining to pre-allocate the storage tensor
                in.seekg(0, std::ios::end);
                std::streamsize f_size = in.tellg();
                in.seekg(8 + header_size, std::ios::beg);
                const auto tensors_size = f_size - 8 - header_size;

                metas = std::unordered_map<std::string, const metadata_t>(metadatas.size());
                // allocate in a way that prevents it from being freed
                storage = new char[tensors_size];
                

                // Read the remaining content
                in.read((char*)storage, static_cast<std::streamsize>(tensors_size));

                // Populate the meta lookup table
                if (metadatas.is_object()) {
                    for (auto &item: metadatas.items()) {
                        if (item.key() != "__metadata__") {
                            const auto name = std::string(item.key());
                            const auto& info = item.value();

                            const metadata_t meta = {info["dtype"].get<TENSORTYPE>(), info["shape"], info["data_offsets"]};
                            metas.insert(std::pair<std::string, safetensors::metadata_t>(name, meta));
                        }
                    }
                }

            }

            };

}

namespace safetensors {

    
    

    
    Tensor<float> safetensors_t::operator[](const char *name) const {
        const auto& meta = metas.at(name);
        char* data_begin = const_cast<char*>(storage) + meta.data_offsets.first;
        // char* data_end = const_cast<char*>(storage.data()) + meta.data_offsets.second;
        if (meta.dtype == TENSORTYPE::kFLOAT_32){
            auto data =  Tensor<float>(meta.shape,(float*)data_begin);
            auto out = Tensor<float>(meta.shape);
            memcpy(out.data, data.data, data.data_size_in_bytes);
            return out;
        }else{
            std::cout << "Unsupported type on getfloat "+std::string(name)+", try .getBfloat16(), data type is "+std::to_string(meta.dtype) << std::endl;
            return Tensor<float>(meta.shape);
            // throw std::runtime_error("Unsupported type on getfloat "+std::string(name)+", try .getBfloat16(), data type is "+std::to_string(meta.dtype));
        }
        
    }

    
    Tensor<bfloat16> safetensors_t::getBF16(const char *name) {
        const auto& meta = metas.at(name);
        char* data_begin = const_cast<char*>(storage) + meta.data_offsets.first;
        // char* data_end = const_cast<char*>(storage.data()) + meta.data_offsets.second;
        if (meta.dtype == TENSORTYPE::kFLOAT_32){
            // throw std::runtime_error("Unsupported type, try []");
            std::cout << "Unsupported type on getBfloat16 "+std::string(name)+", try .getfloat(), data type is "+std::to_string(meta.dtype) << std::endl;
            return Tensor<bfloat16>(meta.shape);
        }
        if (meta.dtype == TENSORTYPE::kBFLOAT_16){
            auto data =  Tensor<bfloat16>(meta.shape,(bfloat16*)data_begin);
            auto out = Tensor<bfloat16>(meta.shape);
            // clone all data from data_begin to out.data
            memcpy(out.data, data.data, data.data_size_in_bytes);
            return out;
        }

        // throw std::runtime_error("Unsupported type, try []");
        std::cout << "Unsupported type on getBfloat16 "+std::string(name)+", try .getfloat(), data type is "+std::to_string(meta.dtype) << std::endl;
        return Tensor<bfloat16>(meta.shape);
    }

    Tensor<uint8_t> safetensors_t::getUCHAR(const char *name) {
        const auto& meta = metas.at(name);
        u_int8_t* data_begin = (u_int8_t*)(storage) + meta.data_offsets.first;
        // char* data_end = const_cast<char*>(storage.data()) + meta.data_offsets.second;
        if (meta.dtype == TENSORTYPE::kFLOAT_32){
            // throw std::runtime_error("Unsupported type, try []");
            std::cout << "Unsupported type on getChar "+std::string(name)+", try .getfloat(), data type is "+std::to_string(meta.dtype) << std::endl;
            // return Tensor<uint8_t>(meta.shape);
        }
        if (meta.dtype == TENSORTYPE::kBFLOAT_16){
            // throw std::runtime_error("Unsupported type on getChar, try .getBfloat16()");
            std::cout << "Unsupported type on getChar "+std::string(name)+", try .getBfloat16(), data type is "+std::to_string(meta.dtype) << std::endl;
            // return Tensor<uint8_t>(meta.shape);
        }
        if (meta.dtype == TENSORTYPE::kUINT_8){
            auto data =  Tensor<uint8_t>(meta.shape,data_begin);
            // auto out = Tensor<uint8_t>(meta.shape);
            // clone all data from data_begin to out.data
            // std::cout << "data size in bytes: " << data.data_size_in_bytes << std::endl;
            // std::cout << "shape: " << data.shape[0] << ":" << data.shape[1] << std::endl;
            // memcpy(out.data, data.data, out.data_size_in_bytes);
            return data;
        }

        // throw std::runtime_error("Unsupported type, try []");
        std::cout << "Unsupported type on getChar "+std::string(name)+", try .getfloat(), data type is "+std::to_string(meta.dtype) << std::endl;
        auto out = Tensor<uint8_t>(meta.shape);
        return out;
    }

    // Tensor<bfloat16> safetensors_t::operator[](const char *name) const {
    //     const auto& meta = metas.at(name);
    //     char* data_begin = const_cast<char*>(storage.data()) + meta.data_offsets.first;
    //     char* data_end = const_cast<char*>(storage.data()) + meta.data_offsets.second;
    //     return Tensor<bfloat16>(meta.shape,(bfloat16*)data_begin);
    // }
}

#endif //SAFETENSORS_H