import safetensors.torch as st
import torch


# change path to your model
path = "./7B.pth"
model = torch.load(path, "cpu")

from torch.utils.cpp_extension import load
quant_cpp = load(name="quant_cpp", sources=["./quant.cpp"], verbose=True,
        extra_cflags=["-O3", "-march=native", "-fopenmp"  ,"-flto",  "-fopenmp", "-funroll-loops", "-D_GLIBCXX_PARALLEL"])


import inquirer

questions = [
    inquirer.List('mode',
                message="What do you want to do?",
                choices=['Convert to BF16', 'Convert to FP32', 'Convert to Uint8'],
            ),
]
mode = inquirer.prompt(questions)["mode"]
bf16 = mode == "Convert to BF16"
fp32 = mode == "Convert to FP32"
uint8 = mode == "Convert to Uint8"

import cpuinfo
hasbf16 = "avx512_bf16" in cpuinfo.get_cpu_info_json()
avx512 = "avx512" in cpuinfo.get_cpu_info_json()
avx2 = "avx2" in cpuinfo.get_cpu_info_json()
neon = "neon" in cpuinfo.get_cpu_info_json()


import tqdm as tqdm
keys = [*model.keys()]
for key in tqdm.tqdm(keys):
    if model[key].shape.__len__() == 2 and key != "emb.weight" and "time_" not in key:
        
        # bf16 conversion for avx512
        if bf16:
            model[key] = model[key].bfloat16().clone().cpu()
            shape = model[key].shape
            model[key] = model[key].reshape(-1,2,16)[:,[1,0]].reshape(shape)
        else:
            if uint8:
                weight = model[key].t().float().clone().cpu()
                model[key] = (torch.zeros(weight.shape[1],weight.shape[0]).to(torch.uint8))
                model[key+".range"] = (torch.zeros(weight.shape[1],16))
                model[key+".zero"] = (torch.zeros(weight.shape[1],16))
                quant_cpp.quantize_cpu(weight.t().contiguous(), model[key+".range"] , model[key+".zero"],model[key], weight.shape[1], weight.shape[0])
                # model[key] = model[key].t().contiguous().cpu()
            else:
                model[key] = model[key].float().clone().cpu()
    elif model[key].shape.__len__() == 1:
        model[key] = model[key].float().cpu()
    else:
        model[key] = model[key].float().cpu()
    if "decay" in key:
        model[key] = model[key].double().exp().neg().exp().float().cpu()
# create ../build/ if not exists
import os
if not os.path.exists("../build/"):
    os.makedirs("../build/")
st.save_file(model, "../build/model.safetensors")