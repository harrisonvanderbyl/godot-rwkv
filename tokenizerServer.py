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

        print(body)

        tokens = tokenizer.encode(body)

        tokens = [str(x) for x in tokens]

        ret = ","
        ret.join(tokens)
        # set content length
        out = ret.join(tokens).encode('utf-8')
        self.send_header('Content-Length', len(out))

        self.send_response(HTTPStatus.OK)
        self.end_headers()
        self.wfile.write(out)

    def do_PUT(self):
        # Get body
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        body = body.decode('utf-8')
        body = body.strip()

        # array is a list of integers like "1,2,3,4" turn into array
        print(body)

        body = body.split(",")
        body = [int(x) for x in body]

        tokens = tokenizer.decode(body)

        self.send_response(HTTPStatus.OK)

        out = tokens.encode('utf-8')

        # set content length
        self.send_header('Content-Length', len(out))

        self.end_headers()

        print(out)
        self.wfile.write(out)

    # def do_DELETE(self):
    #     # shutdown server
    #     self.send_response(HTTPStatus.OK)
    #     self.end_headers()
    #     self.wfile.write("OK")
    #     exit(0)

    # def do_GET(self):
    #     self.send_response(HTTPStatus.OK)
    #     self.end_headers()
    #     self.wfile.write("OK")

port = int(input("Port:"))
print("listening on port:",port)
httpd = socketserver.TCPServer(('', port), Handler)
httpd.serve_forever()
