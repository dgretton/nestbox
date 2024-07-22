from abc import ABC, abstractmethod

class InstanceIDInterface(ABC):
    @abstractmethod
    def get_id(self) -> str:
        """Retrieve the instance ID."""
        pass

    @abstractmethod
    def set_id(self, new_id: str) -> None:
        """Set the instance ID. This might be used in cases where the ID needs to be explicitly set."""
        pass

    @abstractmethod
    def generate_new_id(self) -> str:
        """Generate a new instance ID. Returns the new ID."""
        pass

