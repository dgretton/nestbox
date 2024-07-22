from abc import ABC, abstractmethod
from typing import Any, Dict, Union
from .communication import ConnectionInterface

class ServerInterface(ABC):
    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def handle_connection(self, connection: ConnectionInterface) -> None:
        pass


class ClientInterface(ABC):
    @abstractmethod
    def connect(self, connection: ConnectionInterface) -> None:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass

    @property
    @abstractmethod
    def connection(self) -> Union[ConnectionInterface, None]:
        pass

    @property
    @abstractmethod
    def connection_info(self) -> Dict[str, Any]:
        pass
