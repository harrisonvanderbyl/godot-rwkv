# RWKV+Godot

## What

### Godot 

The Godot Engine is a free, all-in-one, cross-platform game engine that makes it easy for you to create 2D and 3D games.

### RWKV

RWKV is an RNN with Transformer-level LLM performance, which can also be directly trained like a GPT transformer (parallelizable). And it's 100% attention-free. You only need the hidden state at position t to compute the state at position t+1.

### RWKV-CPP-CUDA

RWKV-CPP-CUDA is a c++/cuda library I created that implements the RWKV inference code in pure cuda. This allows for compiled code with no torch or python dependencies, while allowing the full use of GPU acceleration.
The code implements 8bit inference, allowing for quick and light inference.

### Godot+RWKV

Godot+RWKV is a Godot module that I developed using RWKV-CPP-CUDA, and allows the development of games and programs using RWKV to be developed and distributed using godot, without the need to install complex environments and libraries, for both developers and consumers.

## Why

* I felt I could achieve it
* Its something thats needed to advance the use of AI in consumer devices
* The lols
* Attention, because I didnt get much growing up, and RWKV has none
* ADHD hyperfocus

## Where

[Module Repository](https://github.com/harrisonvanderbyl/godot-rwkv)

[RWKV standalone c++/cuda library](https://github.com/harrisonvanderbyl/rwkv-cpp-cuda)

[Prebuilt Godot Executable](https://github.com/harrisonvanderbyl/godot-rwkv/actions/runs/4816463552)

[Model Converter](https://github.com/harrisonvanderbyl/rwkv-cpp-cuda/tree/main/converter)

[Tokenizer Files](https://github.com/harrisonvanderbyl/rwkv-cpp-cuda/tree/main/include/rwkv/tokenizer/vocab)

[Unconverted Models : 14/7/3/1.5B finetuned on all your favorite instruct datasets, in both chinese and english](https://huggingface.co/BlinkDL/rwkv-4-raven/tree/main)

[Your Will To Live](https://i.redd.it/b39ai2k1acwa1.jpg)

[Rick Astley](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

## How

* Download a model (preconverted models pending)
* Convert the model (requires torch to pack tensors into raw binary)
* Download the tokenizer files
* Create a game in godot
* Distribute the game
* Profit

Example Code:

```python
extends Node2D
var zrkv = GodotRWKV.new()

# Called when the node enters the scene tree for the first time.
func _ready():
	zrkv.loadModel("/path/to/model.bin")
	zrkv.loadTokenizer("/path/to/folder/with/vocab/")
	zrkv.loadContext("Hello, my name is Nathan, and I have been trying to reach you about your cars extended warrenty.")
# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	# number of tokens to generate, temperature, tau
	print(zrkv.forward(5,0.9,0.7))
```

## When

* Pls submit PRs if you want them sooner

Soon:

* Windows support (Just needs some scons magic)
* AMD Support (Just needs some HIPify magic)
* CPU mode (Just needs some ggml)
* CPU offload (needs ggml and effort)
* Preconverted models

Later:

* INT4
