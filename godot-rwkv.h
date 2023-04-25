#ifndef RWKV_GODOT_H
#define RWKV_GODOT_H

#include "core/io/resource.h"
#include "core/object/ref_counted.h"
#include "rwkv.h"


class GodotRWKV : public Resource {
	GDCLASS(GodotRWKV, Resource);

	


public:
	RWKV model = RWKV();
	GPT2Tokenizer* tokenizer;
	int lastToken = 187;

	
	GodotRWKV() {
		
	}

	void loadModel(String path) {
		model.loadFile(std::string(path.utf8().get_data()));
	};

	void loadTokenizer(String path) {
		std::optional<GPT2Tokenizer> tokenizerloader = GPT2Tokenizer::load(std::string((path+"vocab.json").utf8().get_data()),std::string((path+"merges.txt").utf8().get_data()));
		if (tokenizerloader.has_value()) {
			tokenizer = new GPT2Tokenizer(tokenizerloader.value());
		}else{
			std::cout << "Error loading tokenizer" << std::endl;
		}
	};

	void loadContext(String context) {
		
	};

	String forward(int number) {
		std::string output = "";
		for (int i = 0; i < number; i++) {
			model.forward(lastToken);
			lastToken = typical(model.out, 1.0, 0.9);
			output += tokenizer->decode({lastToken});
		}
		return String(output.c_str());
	};

	void resetState() {
		
	};

	int getState() {
		
	};

	void setState(int state) {
		
	};

	protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("loadContext"), &GodotRWKV::loadContext);
		ClassDB::bind_method(D_METHOD("forward"), &GodotRWKV::forward);	
		ClassDB::bind_method(D_METHOD("loadModel"), &GodotRWKV::loadModel);
		ClassDB::bind_method(D_METHOD("resetState"), &GodotRWKV::resetState);
		ClassDB::bind_method(D_METHOD("getState"), &GodotRWKV::getState);
		ClassDB::bind_method(D_METHOD("setState"), &GodotRWKV::setState);
		ClassDB::bind_method(D_METHOD("loadTokenizer"), &GodotRWKV::loadTokenizer);
	}
};

#endif // RWKV_GODOT_H
