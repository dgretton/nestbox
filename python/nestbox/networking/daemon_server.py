import socket
import os
import ssl
from ..interfaces.client_server import ServerInterface

class DaemonServer(ServerInterface):
    def __init__(self, address, daemon):
        super().__init__(address)
        self.address = address
        self.daemon = daemon

    def handle_connection(self, client_socket):
        with client_socket:
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break
                response = self.daemon.process_request(data.decode('utf-8'))
                client_socket.sendall(response.encode('utf-8'))


class UnixSocketDaemonServer(DaemonServer):
    def start(self):
        if os.path.exists(self.address):
            os.remove(self.address)
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.bind(self.address)
        self.socket.listen(5)
        print(f"Unix Socket Server running on {self.address}")
        self.accept_connections()

    def accept_connections(self):
        while True:
            client_socket, _ = self.socket.accept()
            self.handle_connection(client_socket)

    def handle_connection(self, client_socket):
        with client_socket:
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break
                response = self.process_request(data.decode('utf-8'))
                client_socket.sendall(response.encode('utf-8'))

    def stop(self):
        self.socket.close()
        if os.path.exists(self.address):
            os.remove(self.address)


class TCPDaemonServer(DaemonServer):
    def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host, port = self.address.split(':')
        self.socket.bind((host, int(port)))
        self.socket.listen(5)
        print(f"TCP Server running on {self.address}")
        self.accept_connections()

    def accept_connections(self):
        while True:
            client_socket, addr = self.socket.accept()
            # secure_socket = ssl.wrap_socket(client_socket, server_side=True, certfile="server.crt", keyfile="server.key") TODO
            # self.handle_connection(secure_socket)
            self.handle_connection(client_socket)

    def handle_connection(self, client_socket):
        with client_socket:
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break
                response = self.process_request(data.decode('utf-8'))
                client_socket.sendall(response.encode('utf-8'))

    def stop(self):
        self.socket.close()
