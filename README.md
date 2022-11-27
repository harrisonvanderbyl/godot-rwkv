
Unzip/git clone, into the ./modules/ folder of godot 4 repository.

Currently only works for linux, but it should be possible for it to work for windows

When running the created executable, you may need to run it from the terminal like so:

`LD_LIBRARY_PATH=$(pwd)/modules/rwkv/libtorch/lib/ ./bin/godot.linuxbsd.tools.x86_64`

Will be working on fixing this or something soon

create the pre, post, and layer files by using https://github.com/harrisonvanderbyl/rwkv_chatbot
and running the `python3 ./createTorchScriptFile.py` command.

Look in the ./pt/ file.

example godot code: 

```py
extends Node2D
@export var resorce:RWKV

# Called when the node enters the scene tree for the first time.
func _ready():
	resorce.set_preprocess("/full/path/to/pre.pt")
	resorce.set_postprocess("/full/path/to/post.pt")
	resorce.set_layers(["/full/path/to/0.pt"]) # all the layers here, set the batch number high to only create 1 layer
	resorce.set_empty_state([60,768]) # 5*layers, n_embed. You can see the data in the /pt/ subfolder name when you create your files
	var out = resorce.invoke(6) # TODO create a tokenizer function for creating input tokens
	var st = resorce.detokenize([out])
	print("output",st)
	var m = []
	for i in range(100):
		out = resorce.invoke(out)
		st = resorce.detokenize([out])
		m.push_back(out)
	print(resorce.detokenize(m))
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	
	pass
```
