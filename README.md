# RWKV+Godot

## What

### Godot 

The Godot Engine is a free, all-in-one, cross-platform game engine that makes it easy for you to create 2D and 3D games.

### RWKV

RWKV is an RNN with Transformer-level LLM performance, which can also be directly trained like a GPT transformer (parallelizable). And it's 100% attention-free. You only need the hidden state at position t to compute the state at position t+1.

### RWKV-HPP

RWKV-HPP is a c++ header library I created that implements the RWKV inference code in pure c++. This allows for compiled code with no torch or python dependencies.
The code implements 8bit inference, allowing for quick and light inference.

### Godot+RWKV

Godot+RWKV is a Godot module that I developed using RWKV-HPP, and allows the development of games and programs using RWKV to be developed and distributed using godot, without the need to install complex environments and libraries, for both developers and consumers.

## Where

[Module Repository](https://github.com/harrisonvanderbyl/godot-rwkv)

[RWKV standalone c++ header library](https://github.com/harrisonvanderbyl/rwkv.hpp)


