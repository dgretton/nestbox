from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Optional
import uuid
from .instance_id import InstanceIDInterface
from .communication import ConnectionInterface
from .client_server import ClientInterface

class DefaultInstanceID(InstanceIDInterface):
    @staticmethod
    def generate_new_id() -> str:
        return str(uuid.uuid4())

    def __init__(self):
        self._id = self.generate_new_id()

    def get_id(self) -> str:
        return self._id

    def set_id(self, new_id: str) -> None:
        self._id = new_id


class PeerInfo:
    def __init__(self, instance_id: str, connection: ConnectionInterface, metadata: Dict[str, Any]):
        self.instance_id = instance_id
        self.connection = connection
        self.metadata = metadata


class PeerDiscoveryListenerInterface(ABC):
    @abstractmethod
    def on_peer_discovered(self, callback: Callable[[PeerInfo], None]) -> None:
        """Register a callback for when a new peer is discovered."""
        pass

    @abstractmethod
    def on_peer_lost(self, callback: Callable[[str], None]) -> None:
        """Register a callback for when a peer is no longer available."""
        pass


class PeerDiscoveryClientInterface(ClientInterface, PeerDiscoveryListenerInterface):
    def __init__(self, instance_id: Optional[InstanceIDInterface] = None):
        self._instance_id = instance_id or self._create_default_instance_id()

    @staticmethod
    def _create_default_instance_id() -> InstanceIDInterface:
        # This should be implemented in subclasses to return an appropriate default
        raise NotImplementedError("Subclasses should implement this method.")

    @property
    def instance_id(self) -> str:
        return self._instance_id.get_id()

    @abstractmethod
    def start_discovery(self) -> None:
        """Start the peer discovery process."""
        pass

    @abstractmethod
    def stop_discovery(self) -> None:
        """Stop the peer discovery process."""
        pass

    @abstractmethod
    def get_peers(self) -> List[PeerInfo]:
        """Get the current list of discovered peers."""
        pass

    @abstractmethod
    def register_self(self, connection: ConnectionInterface, metadata: Dict[str, Any]) -> None:
        """
        Register this instance as a discoverable peer.
        The peer's ID should be included in the metadata.
        """
        metadata['id'] = self.instance_id

    @abstractmethod
    def unregister_self(self) -> None:
        """Unregister this instance from being discoverable."""
        pass


# Usage, imagining that the LocalNetworkPeerDiscovery class is implemented elsewhere:
# peer_discovery = LocalNetworkPeerDiscovery()  # Uses default instance ID
# or
# custom_id = SomeCustomInstanceID() # inherits from InstanceIDInterface
# peer_discovery = LocalNetworkPeerDiscovery(custom_id)  # Uses custom instance ID
