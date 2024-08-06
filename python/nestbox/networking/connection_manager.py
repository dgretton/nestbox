from ..interfaces.communication import ConnectionConfigInterface
from .tcp_connection import TCPConnection, TCPServerConnection
from .unix_socket_connection import UnixSocketConnection, UnixSocketServerConnection
from typing import Dict, Union

class ConnectionConfig(ConnectionConfigInterface):
    def __init__(self, type: str, **config: Dict[str, Union[str, int]]):
        self._type = type
        known_keys = ['address', 'port', 'socket_path', 'cert_file', 'key_file']
        if any(key not in known_keys for key in config):
            raise ValueError(f"Unknown keys in connection config: {', '.join(set(config.keys()) - set(known_keys))}. Allowed keys are: {', '.join(known_keys)}")
        self.__dict__.update({'_' + key: value for key, value in config.items()})
        self.config = config

    @property
    def type(self) -> str:
        return self._type
    
    @property
    def address(self) -> str:
        return self._address
    
    @property
    def port(self) -> int:
        return self._port
    
    @property
    def socket_path(self) -> str:
        return self._socket_path
    
    @property
    def cert_file(self) -> str:
        return self._cert_file
    
    @property
    def key_file(self) -> str:
        return self._key_file

    def __repr__(self):
        return f"ConnectionConfig(type={self.type}, config={self.config})"


class ConnectionManager:
    def __init__(self):
        pass

    def create_connection(self, config: ConnectionConfig):
        assert isinstance(config, ConnectionConfig), f"ConnectionManager expected ConnectionConfig, got {type(config)}"
        if config.type == 'tcp':
            return TCPConnection(config.address, config.port)
        elif config.type == 'unix_socket':
            return UnixSocketConnection(config.address)
        else:
            raise ValueError(f"Unsupported connection type: {config.type}")

    def create_server_connection(self, config: ConnectionConfig):
        if config.type == 'tcp':
            return TCPServerConnection(config.address, config.port)
        elif config.type == 'unix_socket':
            return UnixSocketServerConnection(config.address)
        else:
            raise ValueError(f"Unsupported server connection type: {config.type}")
