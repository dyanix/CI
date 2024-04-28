from xmlrpc.server import SimpleXMLRPCServer

def calculate_factorial(n):
    if n == 0:
        return 1
    else:
        return n * calculate_factorial(n - 1)

server = SimpleXMLRPCServer(("localhost", 8000))
print("Server listening on port 8000...")

server.register_function(calculate_factorial, "factorial")

server.serve_forever()
