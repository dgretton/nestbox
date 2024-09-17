import socket
from ..interfaces.communication import ConnectionInterface, ServerConnectionInterface

class UnixSocketConnection(ConnectionInterface):
    def __init__(self, socket_path):
        self.socket_path = socket_path
        self.socket = None
        
    def connect(self):
        if not self.socket:
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket.connect(self.socket_path)

    def disconnect(self):
        if self.socket:
            self.socket.close()
            self.socket = None

    def send(self, data: bytes):
        self.socket.sendall(data)

    def receive(self):
        return self.socket.recv(4096)

    def is_connected(self):
        return self.socket is not None and self.socket.fileno() != -1
   
    @property
    def connection_info(self):
        return {'type': 'unix_socket', 'address': self.socket_path}


class UnixSocketServerConnection(ServerConnectionInterface):
    def __init__(self, socket_path):
        self.socket_path = socket_path
        self.socket = None
        self.client_sockets = []

    def connect(self):
        if not self.socket:
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket.bind(self.socket_path)
            self.socket.listen(1)
            print(f"Server listening on {self.socket_path}")

    def disconnect(self):
        for client_socket in self.client_sockets:
            client_socket.close()
        self.client_sockets = []
        if self.socket:
            self.socket.close()
            self.socket = None

    def is_connected(self):
        return self.socket is not None and self.socket.fileno() != -1

    def accept(self):
        client_socket, addr = self.socket.accept()
        self.client_sockets.append(client_socket)
        return UnixSocketConnection(client_socket)
    
    @property
    def connection_info(self):
        return {'type': 'unix_socket', 'address': self.socket_path}
