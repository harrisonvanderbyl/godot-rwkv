

import torch
import safetensors.torch as st
file = {
    "test":torch.arange(32).bfloat16().clone().cpu(),
    "test2":torch.arange(32).float().clone().cpu(),
}
# interleave test
file["test"] = file["test"].reshape(-1)
st.save_file(file, "test.safetensors")