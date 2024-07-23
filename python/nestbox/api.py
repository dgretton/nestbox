from .api_client import NestboxAPIClient
from .config import DAEMON_CONN_CONFIG
import numpy as np

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = NestboxAPIClient(DAEMON_CONN_CONFIG)
    return _client

def get_transform(cs1, cs2):
    return _get_client().get_transform(cs1, cs2)

def from_cs(cs_name):
    builder = TransformationBuilder()
    return builder.in_cs(cs_name)

def create_coordinate_system(name=None):
    return _get_client().create_coordinate_system(name)

def add_measurements(cs_guid, measurements):
    return _get_client().add_measurements(cs_guid, measurements)

# Other main module API functions...


class OptimizedTransformer:
    def __init__(self, source_cs, target_cs, time=None):
        self.source_cs = source_cs
        self.target_cs = target_cs
        self.time = time
        self.transform_matrix = self._compute_transform_matrix()

    def _compute_transform_matrix(self):
        # Actual implementation to compute the transformation matrix
        return np.eye(10)  # Placeholder

    def transform(self, state_vector):
        return np.dot(self.transform_matrix, state_vector)

    def transform_many(self, state_vectors):
        return np.dot(state_vectors, self.transform_matrix.T)


class TransformationBuilder:
    def __init__(self):
        self.source_cs = None
        self.target_cs = None
        self.time = None
        self._optimized_transformer = None

    def in_cs(self, cs_name):
        self.source_cs = cs_name
        return self

    def to_cs(self, cs_name):
        self.target_cs = cs_name
        return self

    def at_time(self, t):
        self.time = t
        return self

    def build(self):
        if not self.source_cs or not self.target_cs:
            raise ValueError("Both source and target coordinate systems must be specified")
        self._optimized_transformer = OptimizedTransformer(self.source_cs, self.target_cs, self.time)
        return self

    def transform(self, state_vector):
        if self._optimized_transformer is None:
            self.build()
        return self._optimized_transformer.transform(state_vector)

    def transform_many(self, state_vectors):
        if self._optimized_transformer is None:
            self.build()
        return self._optimized_transformer.transform_many(state_vectors)

    def __call__(self, state_vector):
        return self.transform(state_vector)
