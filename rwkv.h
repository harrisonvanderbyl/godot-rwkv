#ifndef RWKV_H
#define RWKV_H

#include "./decoder.h"
#include "core/io/resource.h"
#include "core/object/ref_counted.h"
// #include "rwkv/lite/model.h"

#include "torch/script.h"
// #include "rwkv/lite/op_resolver.h"
// #include "rwkv/lite/optional_debug_tools.h"
// #include "<iostream>"
#define RWKV_MINIMAL_CHECK(x)                                  \
	if (!(x)) {                                                  \
		fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
		exit(1);                                                 \
	}
// Generator based on a script, like GDScript, C# or NativeScript.
// The script is expected to properly handle multithreading.
class RWKV : public Resource {
	GDCLASS(RWKV, Resource);

private:
	torch::jit::script::Module preprocess;
	torch::jit::script::Module postprocess;
	std::vector<torch::jit::script::Module> layers;
	at::Tensor emptyState;
	at::Tensor currentState;
	// std::unique_ptr<RWKV::Interpreter> interpreter;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_preprocess"), &RWKV::set_preprocess);
		ClassDB::bind_method(D_METHOD("set_postprocess"), &RWKV::set_postprocess);
		ClassDB::bind_method(D_METHOD("set_layers"), &RWKV::set_layers);
		ClassDB::bind_method(D_METHOD("invoke"), &RWKV::invoke);
		ClassDB::bind_method(D_METHOD("set_empty_state"), &RWKV::set_empty_state);
		ClassDB::bind_method(D_METHOD("detokenize"), &RWKV::detokenize);
	}

public:
	void set_preprocess(const String &p_path) {
		const char *c = p_path.utf8().get_data();

		try {
			// Deserialize the ScriptModule from a file using torch::jit::load().
			preprocess = torch::jit::load(c);
			printf("Loaded preprocess model");

		} catch (const c10::Error &e) {
			printf("error loading the model\n");
		}
	}

	void set_postprocess(const String &p_path) {
		const char *c = p_path.utf8().get_data();

		try {
			// Deserialize the ScriptModule from a file using torch::jit::load().
			postprocess = torch::jit::load(c);
			printf("Loaded postprocess model");
		} catch (const c10::Error &e) {
			printf("error loading the model\n");
		}
	}

	void set_layers(const PackedStringArray &p_layers) {
		for (int i = 0; i < p_layers.size(); i++) {
			String p_path = p_layers[i];
			const char *c = p_path.utf8().get_data();

			try {
				// Deserialize the ScriptModule from a file using torch::jit::load().
				layers.push_back(torch::jit::load(c));
				printf("Loaded layer model");
			} catch (const c10::Error &e) {
				printf("error loading the model\n");
			}
		}
	}

	String detokenize(const PackedInt32Array &p_input) {
		String output = "";
		for (int i = 0; i < p_input.size(); i++) {
			output += decoderMap[p_input[i]];
		}
		return output;
	}

	void set_empty_state(const PackedInt32Array &p_state) {
		try {
			// Deserialize the ScriptModule from a file using torch::jit::load().
			emptyState = torch::zeros({ p_state[0], p_state[1] });
			currentState = emptyState.clone();
			printf("Loaded empty state");
		} catch (const c10::Error &e) {
			printf("error loading the model\n");
		}
	}

	int invoke(int num) {
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(at::tensor(num, torch::kInt64));

		// Execute the model and turn its output into a tensor.
		auto output = preprocess.forward(inputs).toTensor();

		print_line("got preprocess output");

		std::vector<torch::jit::IValue> states;

		states.push_back(output);
		states.push_back(currentState);

		for (uint i = 0; i < layers.size(); i++) {
			// Execute the model and turn its output into a tensor.
			auto stateso = layers[i].forward(states).toTuple();
			states = stateso->elements();
		}

		printf("got layers output");

		currentState = states[1].toTensor();

		std::vector<torch::jit::IValue> outputs;
		outputs.push_back(states[0]);

		// Execute the model and turn its output into a tensor.
		at::Tensor output2 = postprocess.forward(outputs).toTensor();

		printf("got postprocess output");

		// Top-k decoding
		auto topk = output2.topk(5);

		at::Tensor topk_indices;
		at::Tensor topk_values;

		std::tie(topk_values, topk_indices) = topk;

		printf("got topk output");

		// print_line("got postprocess output");

		return topk_indices[rand() % 5].item<int>();

		// Top_p sample
	}

	RWKV(){

	};
};

#endif // RWKV_H
