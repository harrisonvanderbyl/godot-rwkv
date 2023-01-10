#ifndef RWKV_H
#define RWKV_H

#include "./decoder.h"
#include "core/io/resource.h"
#include "core/object/ref_counted.h"
// #include "rwkv/lite/model.h"
#include "tokenizer.h"
#include "torch/script.h"

// #include "rwkv/lite/op_resolver.h"
// #include "rwkv/lite/optional_debug_tools.h"
// #include "<iostream>"

// printf("got postprocess output");
// #define RWKV_MINIMAL_CHECK(x)                                    \
// 	if (!(x)) {                                                  \
// 		fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
// 		exit(1);                                                 \
// 	}
// Generator based on a script, like GDScript, C# or NativeScript.
// The script is expected to properly handle multithreading.
class RWKV : public Resource {
	GDCLASS(RWKV, Resource);

private:
	torch::jit::script::Module model;
	Ref<TokenizerServer> tokenizer;
	at::Tensor emptyState;
	at::Tensor currentState;
	int lastToken = 187;

	// std::unique_ptr<RWKV::Interpreter> interpreter;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_model"), &RWKV::set_model);
		ClassDB::bind_method(D_METHOD("forward"), &RWKV::forward);
		ClassDB::bind_method(D_METHOD("load_context"), &RWKV::load_context);
		ClassDB::bind_method(D_METHOD("set_empty_state"), &RWKV::set_empty_state);
		ClassDB::bind_method(D_METHOD("detokenize"), &RWKV::detokenize);
		ClassDB::bind_method(D_METHOD("tokenize"), &RWKV::tokenize);
		ClassDB::bind_method(D_METHOD("get_tokenizer"), &RWKV::get_tokenizer);
		ClassDB::bind_method(D_METHOD("set_tokenizer"), &RWKV::set_tokenizer);
		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "tokenizer", PROPERTY_HINT_RESOURCE_TYPE, "TokenizerServer"), "set_tokenizer", "get_tokenizer");
	}

public:
	void set_tokenizer(Ref<TokenizerServer> inp) {
		tokenizer = inp;
	}
	Ref<TokenizerServer> get_tokenizer() {
		return tokenizer;
	}
	void set_model(const String &p_path) {
		const char *c = p_path.utf8().get_data();

		try {
			// Deserialize the ScriptModule from a file using torch::jit::load().
			model = torch::jit::load(c);
			printf("Loaded model");

		} catch (const c10::Error &e) {
			printf("error the model\n");
		}
	}

	String detokenize(const String &p_input) {
		return tokenizer->req(HTTPClient::METHOD_PUT, p_input);
	}

	void startTokeniserServer(int port) {
	}

	String tokenize(const String &p_input) {
		return tokenizer->req(HTTPClient::METHOD_POST, p_input);
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

	void load_context(String s) {
		auto tokens = tokenize(s);
		auto token_ids = tokens.split(",");
		for (int i = 0; i < token_ids.size(); i++) {
			int token_id = token_ids[i].to_int();
			invoke(token_id);
		}
	}

	int invoke(int num) {
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(at::tensor(num, at::kLong));
		inputs.push_back(currentState);

		// Execute the model and turn its output into a tensor.
		auto output = model.forward(inputs).toTuple();

		// get the tensors from tuple
		auto output_state = output->elements()[1].toTensor();
		auto output_logits = output->elements()[0].toTensor();

		currentState = output_state;

		// printf("got postprocess output");

		// Top-k decoding
		auto topk = output_logits.topk(5);

		at::Tensor topk_indices;
		at::Tensor topk_values;

		std::tie(topk_values, topk_indices) = topk;

		// printf("got topk output");

		// print_line("got postprocess output");
		lastToken = topk_indices[rand() % 5].item<int>();

		return lastToken;

		// Top_p sample
	}

	String forward() {
		return detokenize(String::num_int64(invoke(lastToken)));
	}

	RWKV(){

	};
};

#endif // RWKV_H
