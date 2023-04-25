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
	std::vector<std::tuple<std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>>> states = {};
	
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
		std::vector<long int> tokens = tokenizer->encode(std::string(context.utf8().get_data()));
		for (int i = 0; i < tokens.size(); i++) {
			model.forward(tokens[i]);
		}
		lastToken = tokens[tokens.size()-1];		
	};

	String forward(int number, float temperature = 0.9, float tau = 0.8) {
		std::string output = "";
		for (int i = 0; i < number; i++) {
			model.forward(lastToken);
			lastToken = typical(model.out, temperature, tau);
			output += tokenizer->decode({lastToken});
		}
		return String(output.c_str());
	};

	void resetState() {
		for(int i = 0; i < model.num_embed*model.num_layers; i++) {
			model.stateaa[i] = 0;
			model.statebb[i] = 0;
			model.statedd[i] = 0;
			model.statexy[i] = 0;
			model.statepp[i] = -1e30;
		}
	};

	int getState() {
		std::vector<double> statea = {};
		std::vector<double> stateb = {};
		std::vector<double> stated = {};
		std::vector<double> statex = {};
		std::vector<double> statep = {};

		for(int i = 0; i < model.num_embed*model.num_layers; i++) {
			statea.push_back(model.stateaa[i]);
			stateb.push_back(model.statebb[i]);
			stated.push_back(model.statedd[i]);
			statex.push_back(model.statexy[i]);
			statep.push_back(model.statepp[i]);
		}

		states.push_back(std::make_tuple(statea,stateb,stated,statex,statep));

		return states.size()-1;
	};

	void setState(int state) {
		std::vector<double> statea = std::get<0>(states[state]);
		std::vector<double> stateb = std::get<1>(states[state]);
		std::vector<double> stated = std::get<2>(states[state]);
		std::vector<double> statex = std::get<3>(states[state]);
		std::vector<double> statep = std::get<4>(states[state]);

		for(int i = 0; i < model.num_embed*model.num_layers; i++) {
			model.stateaa[i] = statea[i];
			model.statebb[i] = stateb[i];
			model.statedd[i] = stated[i];
			model.statexy[i] = statex[i];
			model.statepp[i] = statep[i];
		}		
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
