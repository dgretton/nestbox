from abc import abstractmethod
from typing import Any, Dict
from .communication import PubSubInterface
from .client_server import ClientInterface

class VisualizerClientInterface(ClientInterface, PubSubInterface):
    @abstractmethod
    def send_update(self, topic: str, data: Dict[str, Any]) -> None:
        """Send an update to the visualizer on a specific topic."""
        pass
