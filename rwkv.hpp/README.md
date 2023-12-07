### RWKV.hpp

Header only library for cpu inference with rwkv v5

Todos and stuff
- [x] AVX512
- [x] AVX512-skylake
- [x] AVX2
- [x] NEON(Arm)
- [ ] Non-simd
- [x] FP32
- [x] BF16
- [ ] FP16
- [x] INT8
- [ ] Cuda
- [ ] Rocm
- [ ] Vulkan
- [x] Batch Inference
- [x] Sequence Inference ( state generation )
- [x] Static memory usage via buffers
- [ ] Fixing memory leakage
- [x] Example app
- [ ] Windows build .bat
- [ ] Mac build 

### Quickstart

1) go to `./models/`
2) Download a model from https://huggingface.co/BlinkDL/rwkv-5-world/tree/main
3) Edit convert.py to point to the download model
4) run convert.py (your converted model is placed into `./build/`)
5) run `./build.sh`
6) go to `./build`
7) from the terminal, run `./rwkv`