Unzip/git clone, into the ./modules/ folder of godot 4 repository.

Currently only works for linux, but it should be possible for it to work for windows

When running the created executable, you may need to run it from the terminal like so:

`LD_LIBRARY_PATH=$(pwd)/modules/rwkv/libtorch/lib/ ./bin/godot.linuxbsd.tools.x86_64`

Will be working on fixing this or something soon

create the .pt by using https://github.com/harrisonvanderbyl/rwkv_chatbot
and running the `python3 ./runOptimised.py` command, selecting a model, and continuing to use the "export torchscript" option, selecting float32 and cpu for options

The tokenizer is handled by running the tokenizerServer.py file(this is minimalistic, requiring only the transformers library `pip install transformers`, and the 20B_tokenizer.json file).

you will need to set the tokenizer port in the properties menu.

In the future, I hope to bundle them into the c++ package, this way you can export your games without the tokenizer server.

![image](img.png)

example godot code:

```py
extends Node2D

@export var myModel:RWKV
var context = "Hello, I am Roger, the guard here in castle Ensfer, You may not enter."
# Called when the node enters the scene tree for the first time.
func _ready():
	myModel.set_model("/home/harrison/exampleProject/model-24-2048-cpu-torch.float32.pt")
	print("done Loading Model")
	myModel.set_empty_state([24*4,2048]) # note values are similar to filename
	myModel.load_context(context)
	


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	context += myModel.forward()
	$Label.text = context


```


### To Do

1) add gpu support
2) add windows and mac support
3) add android support (should be very possible, and infact be fast with nnapi)