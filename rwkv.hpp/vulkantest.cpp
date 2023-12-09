#include "./include/hvml/tensor.hpp"

#include <iostream>

int main() {

  Tensor<float> aa({512,2048}, 0.5);
  Tensor<uint8_t> aai({512,2048}, 64);


  for (int i = 0; i < 512; i++)
  {
      for (int j = 0; j < 2048; j++)
      {
          int r = rand() % 100;

          aa[i][j].data[0] = float(r) / 100.0;
          aai[i][j].data[0] = r;
      }
      
  }

  Tensor<float> aar({512}, 1.0/100.0);
  Tensor<float> aao({512}, 0.0);



  Tensor<float> output({1,17,512}, 0.0);
  Tensor<float> output2({1,17,512}, 0.0);
  Tensor<float> output3({1,17,512}, 0.0);
  Tensor<float> input({1,17,2048}, 1.0);
  for (int i = 0; i < 1; i++)
  {
      for (int j = 0; j < 15; j++)
      {
          for (int k = 0; k < 2048; k++)
          {
              input[i][j][k].data[0] = rand() % 100;
          }
          
      }
      
  }

  aa.matmul(input, output);
  cudaSetDevice(0);

  std::cout << output << std::endl;

  auto a = aa.sendToVulkan();
  auto b = output2.sendToVulkan();
  auto c = input.sendToVulkan();

  auto vaai = aai.sendToVulkan();
  auto vaar = aar.sendToVulkan();

  auto bb = output3.sendToVulkan();
  auto vaao = aao.sendToVulkan();

  a.matmul(c, b);

  // std::cout << "a: " << a.receiveFromVulkan() << std::endl;
  std::cout << "OUTFLOAT: " << b.receiveFromVulkan() << std::endl;
  // std::cout << "c: " << c.receiveFromVulkan() << std::endl;
  
  vaai.matmul(vaar, vaao, c, bb);

  // std::cout << "aai: " << vaai.receiveFromVulkan() << std::endl;
  std::cout << "OUTINT: " << bb.receiveFromVulkan() << std::endl;
  // std::cout << "c: " << c.receiveFromVulkan() << std::endl;

  // addtest 
    Tensor<float> addtest({1,17,2048}, 1.0);
    Tensor<float> addtest2({1,17,2048}, 1.5);
    Tensor<float> lerptest1({2048}, 0.5);
    Tensor<float> muletes1({1,17,2048}, 0.6);

    Tensor<float> addtest3({1,17,2048}, 0.0);
    Tensor<float> lerptest2({1,17,2048}, 0.0);

    lerptest1.lerp(addtest, addtest2, addtest3);
    muletes1.multiply(2,muletes1);
    std::cout << "lerptest1cpu: " << addtest3.receiveFromVulkan() << std::endl;
    std::cout << "multest1: " << muletes1.receiveFromVulkan() << std::endl;

    addtest.sendToVulkan();
    addtest2.sendToVulkan();
    addtest3.sendToVulkan();
    lerptest1.sendToVulkan();
    lerptest2.sendToVulkan();
    muletes1.sendToVulkan();

    muletes1.multiply(2,muletes1);

    addtest.add(addtest2, addtest3);

    std::cout << "addtest: " << addtest3.receiveFromVulkan() << std::endl;

    lerptest1.lerp(addtest, addtest2, lerptest2);

    std::cout << "lerptest2: " << lerptest2.receiveFromVulkan() << std::endl;

    std::cout << "muletes1: " << muletes1.receiveFromVulkan() << std::endl;

  
  return 0;
  

}
