#include "./include/hvml/tensor.hpp"

#include <iostream>

int main() {

  ulong IN = int(4096*4);
  ulong OUT = int(4096);

  // Tensor<float,HVMLCPU> aa({IN,OUT}, 0.5);
  Tensor<uint8_t,HVMLCPU> aai({IN,OUT}, 1);

//   for (int i = 0; i < IN; i++)
//   {
//       for (int j = 0; j < IN; j++)
//       {
//           int r = (rand() % 100)/8;

//           aai[i][j].data[0] = r;
//       }
      
//   }

  Tensor<float,HVMLCPU> aar({OUT}, 1.0/100.0);
//   for (int i = 0; i < OUT; i++)
//   {
//       aar[i].data[0] += (rand() % 100)/50.0;
//   }
  Tensor<float,HVMLCPU> aao({OUT}, 0.1);
//   for (int i = 0; i < OUT; i++)
//   {
//       aao[i].data[0] += (rand() % 100)/100.0;
//   }

  

  Tensor<float,HVMLCPU> output({1,1,OUT}, 0.0);
  Tensor<float,HVMLCPU> output2({1,1,OUT}, 0.0);
  Tensor<float,HVMLCPU> output3({1,1,OUT}, 0.0);
  Tensor<float,HVMLCPU> output4({1,1,OUT}, 0.0);
  Tensor<float,HVMLCPU> outputa({1,1,OUT}, 0.0);
  Tensor<float,HVMLCPU> outputa2({1,1,OUT}, 0.0);
  Tensor<float,HVMLCPU> outputa3({1,1,OUT}, 0.0);
  Tensor<float,HVMLCPU> outputa4({1,1,OUT}, 0.0);
  Tensor<float,HVMLCPU> input({1,1,IN}, 0.0125);
  Tensor<float,HVMLCPU> outputln = Tensor<float>({1,1,IN}, 0.0);
  
  // for (int i = 0; i < 1; i++)
  // {
  //     for (int j = 0; j < 1; j++)
  //     {
  //         for (int k = 0; k < IN; k++)
  //         {
  //             input[i][j][k].data[0] = (rand() % 100)/400.0;
  //         }
          
  //     }
      
  // }

//   input.layernorm(aar,aao,outputln);
//   std::cout << "outputln: " << outputln << std::endl;

  // aai.matmul(aar, aao, input, output);
  aai.matmul(aar, aao, input, output);

  // std::cout << output << std::endl;
  // start timer 
    auto start = std::chrono::high_resolution_clock::now(); 
    for (int i = 0; i < 100; i++)
    {
        aai.matmul(aar, aao, input, output);
        aai.matmul(aar, aao, input, output2);
        aai.matmul(aar, aao, input, output3);
        aai.matmul(aar, aao, input, output4);
    }
    // stop timer
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "CPU1x4: " << duration.count() << std::endl;

  std::cout << output3 << std::endl;

    // matmul(aai,aar, aao, input, output,aai,aar, aao, input, output,aai,aar, aao, input, output,aai,aar, aao, input, output);

  // std::cout << output << std::endl;
  // start timer 
   
    // stop timer
    // stop = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "CPU-4: " << duration.count() << std::endl;

  std::cout << outputa3 << std::endl;

  // auto deviceProp = cudaDeviceProp();
  // cudaGetDeviceProperties(&deviceProp, 0);
  // cudaSetDevice(0);
  // std::cout << "Device: " << deviceProp.name << std::endl;


//   auto a = aa.sendToVulkan();
//   auto b = output2.sendToVulkan();
//   auto c = input.sendToVulkan();

//   auto vaai = aai.sendToVulkan();
//   auto vaar = aar.sendToVulkan();

//   auto bb = output3.sendToVulkan();
//   auto cc = output4.sendToVulkan();
//   auto vaao = aao.sendToVulkan();

//   auto lnout = outputln.sendToVulkan();

//   c.layernorm(vaar,vaao,lnout);
//   // std::cout << "outputln: " << outputln << std::endl;

  



//   vaai.matmul(vaar, vaao, c, bb);

//   // std::cout << "aai: " << vaai.receiveFromVulkan() << std::endl;
//   std::cout << "OUTINT: " << bb.receiveFromVulkan() << std::endl;
//   // std::cout << "c: " << c.receiveFromVulkan() << std::endl;

//     start = std::chrono::high_resolution_clock::now();

//     for (int i = 0; i < 20; i++)
//     {
//         vaai.matmul(vaar, vaao, c, cc);
//     }

//     stop = std::chrono::high_resolution_clock::now();

//     duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

//     std::cout << "GPU: " << duration.count() << std::endl;


    // Tensor<float,HVMLCPU> testing1({1,1,OUT}, 1.0);
    // Tensor<float,HVMLCPU> testing2({1,1,OUT}, 2.0);
    // Tensor<float,HVMLCPU> testing3({1,1,OUT}, 0.0);
    // Tensor<float,HVMLCPU> testing4({1,1,OUT}, 0.0);
    // Tensor<float,HVMLCPU> testing5({1,1,OUT}, 0.0);

    // // time testing
    // auto start = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < 2000; i++)
    // {
    //     testing1.add(testing2, testing1);
    // }
    // auto stop = std::chrono::high_resolution_clock::now();

    // std::cout << testing1 << std::endl;

    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);



    // std::cout << "CPU ADD: " << duration.count() << std::endl;

    // auto ta = testing1.sendToVulkan();
    // auto tb = testing2.sendToVulkan();
    // auto tc = testing3.sendToVulkan();
    // ta.add(tb, tc);
    // ta.add(tb, tc);
    // ta.add(tb, tc);
    // start = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < 2000; i++)
    // {
    //     ta.add(tb, tc);
    // }
    // stop = std::chrono::high_resolution_clock::now();

    // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // std::cout << tc << std::endl;

    // std::cout << "GPU ADD: " << duration.count() << std::endl;


  return 0;
  

}
