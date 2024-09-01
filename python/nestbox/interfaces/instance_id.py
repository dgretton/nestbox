from abc import ABC, abstractmethod

"""Interface for managing instance IDs, especially the instance ID of the current process."""

class InstanceIDInterface(ABC):
    @abstractmethod
    def get_id(self) -> str:
        """Retrieve the persistent instance ID of the current nestbox instance, most likely from disk or database."""
        pass

    @abstractmethod
    def set_id(self, new_id: str) -> None:
        """Set the instance ID. This might be used in cases where the ID needs to be explicitly set e.g. for testing or recovery."""
        pass

    @abstractmethod
    def generate_new_id(self) -> str:
        """Generate a new instance ID, e.g. first-time startup. Returns the new ID."""
        pass

