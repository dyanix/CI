import xmlrpc.client

server = xmlrpc.client.ServerProxy("http://localhost:8000/")

def get_factorial(n):
    return server.factorial(n)


def callServer():
    num = int(input("Enter a number to calculate its factorial: "))
    result = get_factorial(num)
    print(f"The factorial of {num} is {result}")

callServer()



# if __name__ == "__main__":
#     num = int(input("Enter a number to calculate its factorial: "))
#     result = get_factorial(num)
#     print(f"The factorial of {num} is {result}")

'''
    Client Servre
    ComA ComB Tralation
    Marshaling UnMarshaling
    Interface defined lang
    Sync, Async
    Fault tolerance
    TCP/UDP/HTTP

'''