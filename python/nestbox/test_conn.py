import socket
from nestbox.protos import Twig

def receive_full_message(conn):
    # Read the length of the message (first 4 bytes)
    length_data = conn.recv(4)
    print(length_data)
    if not length_data:
        return None
    message_length = int.from_bytes(length_data, byteorder='big') # RFC1700 says network byte order is big-endian
    print(message_length)

    # Read the message data based on the length
    data = b''
    while len(data) < message_length:
        print(f"message_length: {message_length}, len(data): {len(data)}, diff: {message_length - len(data)}")
        packet = conn.recv(message_length - len(data))
        if not packet:
            return None
        data += packet
    return data

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('10.1.20.59', 12345))
#server_socket.bind(('127.0.0.1', 12345))
print(f"Server is bound to {server_socket.getsockname()}")
server_socket.listen(1)
print("Server is listening...")
connection, addr = server_socket.accept()
print(f"Connected by {addr}")

try:
    while True:
        data = receive_full_message(connection)
        if data is None:
            break
        try:
            twig = Twig(data)
            print("Received Twig:")
            print(twig)
        except Exception as e:
            # full traceback
            import traceback
            traceback.print_exc()
            print(f"Error: {e}")
finally:
    connection.close()