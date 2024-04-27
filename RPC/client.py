import xmlrpc.client

def main():
    proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")
    number = int(input("Enter an integer to calculate its factorial:"))


    try:
        result = proxy.calculate_factorial(number)
        print(f"The factorial of {number} is {result}")
    except Exception as e:
        print("Error:",e)


if __name__ == "__main__":
    main()