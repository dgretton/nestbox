from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

class PublisherInterface(ABC):
    @abstractmethod
    def publish(self, topic: str, message: Any) -> None:
        pass

class SubscriberInterface(ABC):
    @abstractmethod
    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        pass

    @abstractmethod
    def register_callback(self, topic: str, callback: Callable[[Any], None]) -> None:
        """Register a callback for messages from the visualizer on a specific topic."""
        pass

    @abstractmethod
    def unregister_callback(self, topic: str, callback: Callable[[Any], None]) -> None:
        """Unregister a callback for a specific topic."""
        pass

    @abstractmethod
    def unsubscribe(self, topic: str) -> None:
        pass


class PubSubInterface(PublisherInterface, SubscriberInterface):
    pass


class ConnectionInterface(ABC):
    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def send(self, data: bytes) -> None:
        pass

    @abstractmethod
    def receive(self) -> bytes:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass

    @property
    @abstractmethod
    def connection_info(self) -> Dict[str, Any]:
        """
        Return a dictionary with connection information.
        This could include things like address, protocol, etc.
        """
        pass
