from abc import abstractmethod
from typing import Any, Dict, List
from .communication import PubSubInterface
from .client_server import ClientInterface

# interface for alignment return data
class AlignmentResultInterface:
    @property
    @abstractmethod
    def timestamp(self) -> str:
        pass

    @property
    @abstractmethod
    def status(self) -> str:
        pass

    @property
    @abstractmethod
    def matrix_transform(self) -> List[List[float]]:
        pass

    @property
    @abstractmethod
    def origin(self) -> List[float]:
        pass

    @property
    @abstractmethod
    def quaternion(self) -> List[float]:
        pass

    @property
    @abstractmethod
    def delta_velocity(self) -> List[float]:
        pass

    @abstractmethod
    def to_json(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, data: Dict[str, Any]) -> 'AlignmentResultInterface':
        pass


class AlignerClientInterface(ClientInterface, PubSubInterface):
    @abstractmethod
    def create_coordinate_system(self, cs_guid: str) -> None:
        """Create a new coordinate system with the given guid."""
        pass

    def add_twig(self, twig: bytes) -> None:
        """Add a new twig (multiple binary-encoded measurement sets) to the given coordinate system."""
        pass

    def add_measurement_set(self, cs_guid: str, dimensions: List[str], samples: List[Dict[str, Any]], transform: List[List[float]], is_homogenous: List[bool]) -> None:
        """Add a new measurement set to the given coordinate system."""
        pass

    @abstractmethod
    def start_alignment(self) -> None:
        """Request the aligner to perform an alignment with the current configuration."""
        pass

    @abstractmethod
    def cancel_alignment(self) -> None:
        """Request the aligner to cancel the current alignment process."""
        pass

    @abstractmethod
    def get_alignment_status(self) -> Dict[str, Any]:
        """Get the current status of the alignment process."""
        pass

    @abstractmethod
    def get_cs_status(cs_guids: List[str]) -> Dict[str, Any]:
        """Get the current alignment status of given coordinate systems."""
        pass

    @abstractmethod
    def get_coordinate_systems(self) -> Dict[str, Any]:
        """Get a list of guids of coordinate systems in the aligner."""
        pass

    @abstractmethod
    def get_basis_change_transform(self, source_cs_guid: str, target_cs_guid: str) -> AlignmentResultInterface:
        """Get the transform to express vectors in one coordinate system in the basis of another."""
        pass

    @abstractmethod
    def set_alignment_parameters(self, params: Dict[str, Any]) -> None:
        """Set parameters for the alignment process."""
        pass

    @abstractmethod
    def delete_coordinate_system(self, cs_guid: str) -> None:
        """Delete the coordinate system with the given guid."""
        pass

    @abstractmethod
    def pin_coordinate_system(self, cs_guid: str) -> None:
        """Pin the given coordinate system to the origin."""
        pass

    def unpin(self) -> None:
        """Unpin the currently pinned coordinate system."""
        pass
