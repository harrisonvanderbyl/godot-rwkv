Unzip/git clone, into the ./modules/ folder of godot 4 repository.

Currently only works for linux, but it should be possible for it to work for windows

When running the created executable, you may need to run it from the terminal like so:

`LD_LIBRARY_PATH=$(pwd)/modules/rwkv/libtorch/lib/ ./bin/godot.linuxbsd.tools.x86_64`

Will be working on fixing this or something soon

create the pre, post, and layer files by using https://github.com/harrisonvanderbyl/rwkv_chatbot
and running the `python3 ./createTorchScriptFile.py` command.

Look in the ./pt/ file.

The tokenizer is handled by running the tokenizerServer.py file(this is minimalistic, requiring only the transformers library `pip install transformers`, and the 20B_tokenizer.json file).

you will need to set the tokenizer port in the properties menu.

In the future, I hope to bundle them into the c++ package, this way you can export your games without the tokenizer server.

![image](img.png)

example godot code:

```py
extends Node2D
@export var resorce:RWKV

# Called when the node enters the scene tree for the first time.
func _ready():
	resorce.set_preprocess("/home/harrison/dabloonn/pre.pt")
	resorce.set_postprocess("/home/harrison/dabloonn/post.pt")
	resorce.set_layers(["/home/harrison/dabloonn/0.pt"])
	resorce.set_empty_state([60,768])
	var prompt = "The following is a conversation between a highly knowledgeable and intelligent AI assistant, called RWKV, and a human user, called User. In the following interactions, User and RWKV will converse in natural language, and RWKV will do its best to answer User’s questions. RWKV was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. The conversation begins.

User: OK RWKV, I’m going to start by quizzing you with a few warm-up questions. Who is currently the president of the USA?

RWKV: It’s Joe Biden; he was sworn in earlier this year.

User: What year was the French Revolution?

RWKV: It started in 1789, but it lasted 10 years until 1799.

User: Can you guess who I might want to marry?

RWKV: Only if you tell me more about yourself - what are your interests?

User: Aha, I’m going to refrain from that for now. Now for a science question. What can you tell me about the Large Hadron Collider (LHC)?

RWKV: It’s a large and very expensive piece of science equipment. If I understand correctly, it’s a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

User: Whats is the LHC looking for?

RWKV: "

	var tokenso = resorce.tokenize(prompt)
	print("attempt")
	print(tokenso)
	var tokens:Array[int] = []

	for t in tokenso.split(","):
		tokens.push_back(t.to_int())
	var out
	for i in tokens:
		out = resorce.invoke(i)
	print("finished preprocess")
	var m = "187"
	for i in range(100):
		out = resorce.invoke(out)


		m = m +","+ str(out)
	var s = ""
	print("finished invoke")
	print(resorce.detokenize(m))
	print("hmm")
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):

	pass

```
