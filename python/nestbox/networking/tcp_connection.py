import socket
from ..interfaces.communication import ConnectionInterface, ServerConnectionInterface

class TCPConnection(ConnectionInterface):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        if not self.socket:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")

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
        return {'type': 'tcp',
                'host': self.host,
                'port': self.port,
                'address': f"{self.host}:{self.port}"}
    

class TCPServerConnection(ServerConnectionInterface):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None
        self.client_sockets = []

    def connect(self):
        if not self.socket:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            print(f"Server listening on {self.host}:{self.port}")

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
        return TCPConnection(addr[0], addr[1])
