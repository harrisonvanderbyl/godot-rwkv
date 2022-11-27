from http import HTTPStatus
import socketserver
import http.server
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="20B_tokenizer.json")


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        # Get body
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        body = body.decode('utf-8')
        body = body.strip()

        tokens = tokenizer.encode(body)

        self.send_response(HTTPStatus.OK)
        self.end_headers()
        self.wfile.write(str(tokens).encode('utf-8'))

    def do_PUT(self):
        # Get body
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        body = body.decode('utf-8')
        body = body.strip()

        # array is a list of integers like "1,2,3,4" turn into array

        body = body.split(",")
        body = [int(x) for x in body]

        tokens = tokenizer.decode(body)

        self.send_response(HTTPStatus.OK)
        self.end_headers()
        self.wfile.write(tokens.encode('utf-8'))

    def do_DELETE(self):
        # shutdown server
        self.send_response(HTTPStatus.OK)
        self.end_headers()
        self.wfile.write("OK")
        exit(0)

    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()
        self.wfile.write("OK")


httpd = socketserver.TCPServer(('', 8999), Handler)
httpd.serve_forever()
