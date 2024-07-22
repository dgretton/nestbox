from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

class MeasurementInterface(ABC):
    @abstractmethod
    def store_measurement(self, measurement: Dict[str, Any]) -> str:
        """Store a new measurement and return its ID."""
        pass

    @abstractmethod
    def get_measurement(self, measurement_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a measurement by its ID."""
        pass

    @abstractmethod
    def get_measurements(self, filter_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve measurements based on filter parameters."""
        pass


class AlignmentHistoryInterface(ABC):
    @abstractmethod
    def store_alignment_result(self, alignment: Dict[str, Any]) -> str:
        """Store a new alignment result and return its ID."""
        pass

    @abstractmethod
    def get_alignment_result(self, alignment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an alignment result by its ID."""
        pass

    @abstractmethod
    def get_alignment_history(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Retrieve alignment history within a specified time range."""
        pass


class FeatureFileInterface(ABC):
    @abstractmethod
    def store_feature_file(self, file_data: bytes, metadata: Dict[str, Any]) -> str:
        """Store a new feature file and return its ID."""
        pass

    @abstractmethod
    def get_feature_file(self, file_id: str) -> Optional[bytes]:
        """Retrieve a feature file by its ID."""
        pass

    @abstractmethod
    def get_feature_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a feature file by its ID."""
        pass


class QueryInterface(ABC):
    @abstractmethod
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a custom query on the database."""
        pass


class DatabaseInterface(MeasurementInterface, AlignmentHistoryInterface, FeatureFileInterface, QueryInterface):
    # Example usage of transaction methods:
    # try:
    #     db.begin_transaction()
    #     # Perform multiple database operations
    #     db.store_measurement(measurement1)
    #     db.store_alignment_result(alignment1)
    #     db.commit_transaction()
    # except Exception as e:
    #     db.rollback_transaction()
    #     # Handle the error

    @abstractmethod
    def connect(self) -> None:
        """Establish a connection to the database."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection to the database."""
        pass

    @abstractmethod
    def begin_transaction(self) -> None:
        """Begin a new transaction."""
        pass

    @abstractmethod
    def commit_transaction(self) -> None:
        """Commit the current transaction."""
        pass

    @abstractmethod
    def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        pass
    