#ifndef RWKV_H
#define RWKV_H

#include "core/io/resource.h"
#include "core/object/ref_counted.h"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

namespace py = pybind11;
using namespace std;
using namespace py::literals;
py::scoped_interpreter guard{};
class RWKV : public Resource {
	GDCLASS(RWKV, Resource);




public:

	py::object model;
	std::vector<py::object> states;
	RWKV() {}
	void loadModel(String path) {
		py::module load = py::module::import("rwkvstic.load");
		py::object pathstring = py::cast(path.utf8().get_data());
		model = load.attr("RWKV")("path"_a = pathstring);
	};

	void loadContext(String context) {
		py::object contextString = py::cast(context.utf8().get_data());
		model.attr("loadContext")("newctx"_a = contextString, "batch"_a = 1);
	};

	String forward(int number) {
		return String(model.attr("forward")("number"_a = number).attr("get")("output").cast<std::string>().c_str());
	};

	void resetState() {
		model.attr("resetState")();
	};

	int getState() {
		states.push_back(model.attr("getState")());
		return states.size() - 1;
	};

	void setState(int state) {
		model.attr("setState")("state"_a = states[state]);
	};



	protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("loadContext"), &RWKV::loadContext);
		ClassDB::bind_method(D_METHOD("forward"), &RWKV::forward);	
		ClassDB::bind_method(D_METHOD("loadModel"), &RWKV::loadModel);
		ClassDB::bind_method(D_METHOD("resetState"), &RWKV::resetState);
		ClassDB::bind_method(D_METHOD("getState"), &RWKV::getState);
		ClassDB::bind_method(D_METHOD("setState"), &RWKV::setState);
	}
};

#endif // RWKV_H
