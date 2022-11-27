#ifndef TokenizerServer_H
#define TokenizerServer_H

#include "core/io/http_client.h"
#include "core/io/resource.h"
#include "core/object/ref_counted.h"

class TokenizerServer : public Resource {
	GDCLASS(TokenizerServer, Resource);

private:
	int port = 8000;
	HTTPClient *connection;
	// std::unique_ptr<TokenizerServer::Interpreter> interpreter;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("req"), &TokenizerServer::req);
		ClassDB::bind_method(D_METHOD("set_port"), &TokenizerServer::set_port);
		ClassDB::bind_method(D_METHOD("get_port"), &TokenizerServer::get_port);

		ADD_PROPERTY(PropertyInfo(Variant::INT, "port"), "set_port", "get_port");
	}

public:
	void set_port(int inp) {
		port = inp;
	}

	int get_port() { return port; }

	String req(HTTPClient::Method method, String data = "", String url = "localhost") {
		connection = HTTPClient::create();

		auto err = connection->connect_to_host(url, port);

		if (err != OK) {
			print_line("Cannot connect");
		}

		while (connection->get_status() == HTTPClient::STATUS_CONNECTING or connection->get_status() == HTTPClient::STATUS_RESOLVING) {
			connection->poll();
		}

		err = connection->request(method, "/", PackedStringArray(), data.to_ascii_buffer().ptr(), data.length());

		while (connection->get_status() == HTTPClient::STATUS_REQUESTING) {
			connection->poll();
		}

		String s = "";

		while (connection->get_status() == HTTPClient::STATUS_BODY) {
			connection->poll();
			auto chunk = connection->read_response_body_chunk();
			if (chunk.size() > 0) {
				if (chunk.size() > 0) {
					const uint8_t *r = chunk.ptr();
					CharString cs;
					cs.resize(chunk.size() + 1);
					memcpy(cs.ptrw(), r, chunk.size());
					cs[chunk.size()] = 0;

					s = s + cs.get_data();
				}
			}
		}

		return s;

		// auto m = OS();
		// OS::execute("wget", [ "-qO-", "https://raw.githubusercontent.com/harrisonvanderbyl/godot-rwkv/master/tokenizerServer.py", "|", "python3", "-" ])
	}

	TokenizerServer(){

	};
};

#endif // TokenizerServer_H
