
#include "safetensors/safetensors.hpp"
#include <fstream>  
int main(){
    std::ifstream inFile;
    inFile.open("./test.safetensors", std::ios::binary);
    auto model = safetensors::deserialize(inFile);
    std::cout << "Model loaded: " << std::endl;
    std::cout << model.keys()[0] << model.keys()[1] << std::endl;
    auto a = model.getBF16("test");
    auto b = model["test2"];
    auto c = LOADFP32BF16(b.data);



    for (int i = 0; i < 32; i++)
    {
        ushort temp = a.data[i];
        uint32_t temp2 = temp;
        temp2 = temp2 << 16;
        std::cout << *(float*)&temp2 << std::endl;
        // a.data[i] = temp;

    }

    __m256i aa = *(__m256i*)a.data;
    __m256i ab = *(__m256i*)(a.data + 16);

    std::cout << ((short*)&aa)[0] << "," << ((short*)&aa)[1] << "," << ((short*)&aa)[2] << std::endl;
    std::cout << ((short*)&ab)[0] << "," << ((short*)&ab)[1] << "," << ((short*)&ab)[2] << std::endl;

    __m512i ac = _mm512_cvtepi16_epi32(aa);
    __m512i ad = _mm512_cvtepi16_epi32(ab);

    std::cout << ((uint32_t*)&ac)[0] << "," << ((uint32_t*)&ac)[1] << "," << ((uint32_t*)&ac)[2] << std::endl;
    std::cout << ((uint32_t*)&ad)[0] << "," << ((uint32_t*)&ad)[1] << "," << ((uint32_t*)&ad)[2] << std::endl;

    // bit shift 16 bits to left
    __m512i ae = _mm512_slli_epi32(_mm512_cvtepi16_epi32(*(__m256i*)a.data), 16);
    __m512i af = _mm512_slli_epi32(ad, 16);

    std::cout << ((float*)&ae)[0] << "," << ((float*)&ae)[1] << "," << ((float*)&ae)[2] << std::endl;
    std::cout << ((float*)&af)[0] << "," << ((float*)&af)[1] << "," << ((float*)&af)[2] << std::endl;

    // for (int i = 0; i < 32; i++)
    // {
    //     ushort temp = c[i];
    //     uint32_t temp2 = temp;
    //     temp2 = temp2 << 16;
    //     std::cout << *(float*)&temp2 << std::endl;
    //     // a.data[i] = temp;
    // }

    // auto aa = *(__m512bh*)a.data;

    // std::cout << (aa)[0] << "," << aa[1] << "," << aa[2] << std::endl;
    // std::cout << b << std::endl;
    // std::cout << c[0] << "," << c[1] << "," << c[2] << std::endl;
}