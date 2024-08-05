from .api_client import NestboxAPIClient
from .config import DAEMON_CONN_CONFIG
from .protos import Dim
from .numutil import coerce_numpy, coerce_quaternion
from .interfaces import AlignmentResultInterface
import numpy as np

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = NestboxAPIClient(DAEMON_CONN_CONFIG)
    return _client

def transform(source_cs, target_cs, type='convert'):
    return TransformationBuilder().in_cs(source_cs).to_cs(target_cs).using_relation(type)

def from_cs(cs_name):
    builder = TransformationBuilder()
    return builder.in_cs(cs_name)

def create_coordinate_system(name=None):
    return _get_client().create_coordinate_system(name)

def add_measurement(feature, cs, mean, covariance, dimensions=None, is_homogeneous=None):
    if dimensions is None:
        dimensions = Dim.all[:len(mean)]
    if is_homogeneous is None:
        is_homogeneous = [False] * len(mean)
    print(f"Adding measurement with dimensions {dimensions} and is_homogeneous {is_homogeneous}")
    return _get_client().add_normal_measurement(feature, cs, mean, covariance, dimensions, is_homogeneous)

def add_measurements(cs_guid, measurements):
    return _get_client().add_measurements(cs_guid, measurements)

def start_alignment():
    return _get_client().start_alignment()

# Other main module API functions...


class TransformationBuilder:
    def __init__(self):
        self.source_cs = None
        self.target_cs = None
        self.components = [Dim.X, Dim.Y, Dim.Z]
        # self.defaults = {dim: None for dim in self.components}
        # self.defaults.update({Dim.VX: 0.0, Dim.VY: 0.0, Dim.VZ: 0.0})
        self.defaults = {Dim.VX: 0.0, Dim.VY: 0.0, Dim.VZ: 0.0}
        self.relation_type = None
        self._optimized_transformer = None

    def in_cs(self, cs_name):
        self.source_cs = cs_name
        return self

    def to_cs(self, cs_name):
        self.target_cs = cs_name
        return self

    def using_relation(self, relation_type):
        self.relation_type = relation_type
        return self

    def with_defaults(self, **defaults):
        self.defaults.update(defaults)
        return self

    def for_components(self, *components):
        self.components = components
        return self

    def at_time(self, t):
        # Convenience method to add time component
        self.defaults[Dim.T] = t
        return self

    def now(self):
        import time
        return self.at_time(time.time()) #TODO: should set a flag or property to indicate that the current time should be fetched from the aligner or calculated based on the response timestamp

    def build(self):
        if not self.source_cs or not self.target_cs or not self.components or not self.relation_type:
            raise ValueError("Source CS, target CS, components, and relation type must be set before building")
        transform_response = _get_client().get_transform(self.source_cs, self.target_cs, relation_type=self.relation_type)
        self._optimized_transformer = OptimizedTransformer(self.components, transform_response, self.defaults)
        return self

    def transform(self, state_vector):
        if self._optimized_transformer is None:
            self.build()
        return self._optimized_transformer.transform(state_vector)

    def transform_many(self, state_vectors):
        if self._optimized_transformer is None:
            self.build()
        return self._optimized_transformer.transform_many(state_vectors)

    def convert(self, state_vector):
        return self.using_relation('convert').transform(state_vector)

    def convert_many(self, state_vectors):
        return self.using_relation('convert').transform_many(state_vectors)


class OptimizedTransformer:
    TRANSFORM_DIMS = [Dim.X, Dim.Y, Dim.Z, Dim.T]
    N = len(TRANSFORM_DIMS)
    POS_IDX = TRANSFORM_DIMS.index(Dim.X)
    T_IDX = TRANSFORM_DIMS.index(Dim.T)
    #V_IDX = TRANSFORM_DIMS.index(Dim.VX)

    def __init__(self, dims, transform_response, defaults=None):
        assert isinstance(transform_response, AlignmentResultInterface)
        origin = transform_response.origin
        quaternion = transform_response.quaternion
        delta_velocity = transform_response.delta_velocity
        if not all(dv == 0.0 for dv in delta_velocity):
            raise NotImplementedError("Nonzero delta velocity not yet supported")
        if defaults is None:
            self.defaults = {}
        else:
            self.defaults = defaults
        self.origin = coerce_numpy(origin)
        assert self.origin.shape == (3,)
        self.quaternion = coerce_quaternion(quaternion)
        self.delta_velocity = coerce_numpy(delta_velocity)
        assert self.delta_velocity.shape == (3,)
        self.dims = dims
        self._permutation_matrix = self._create_permutation_matrix()
        self._defaults_matrix = self._create_defaults_matrix()
        self._rotation_matrix = self._create_rotation_matrix()

    def _create_permutation_matrix(self):
        matrix = np.zeros((self.N, self.N))
        for i, dim in enumerate(self.TRANSFORM_DIMS):
            if dim in self.dims:
                # If the dimension is in the input, set the corresponding row
                j = self.dims.index(dim)
                matrix[i, j] = 1.0
            else:
                matrix[i, i] = 1.0
        return matrix

    def _create_rotation_matrix(self):
        rotation_matrix_3d = self.quaternion.rotation_matrix
        rotation_matrix = np.eye(self.N)
        rotation_matrix[self.POS_IDX:self.POS_IDX+3, self.POS_IDX:self.POS_IDX+3] = rotation_matrix_3d
        #rotation_matrix[self.V_IDX:self.V_IDX+3, self.V_IDX:self.V_IDX+3] = rotation_matrix_3d
        return rotation_matrix

    def _create_defaults_matrix(self):
        matrix = np.eye(self.N)
        for i, dim in enumerate(self.TRANSFORM_DIMS):
            if dim in self.defaults:
                matrix[i, i] = self.defaults[dim]
        return matrix

    def transform(self, state_vector):
        state_vector = coerce_numpy(state_vector)
        return self.transform_many(state_vector[np.newaxis])[0]

    def transform_many(self, state_vectors):
        state_vectors = coerce_numpy(state_vectors)
        if state_vectors.shape[1] != len(self.dims):
            raise ValueError(f"Input state vectors must have {len(self.dims)} components ({', '.join((Dim.name(d) for d in self.dims))})")
        # TODO: we go to the trouble of reorganizing the matrix so much in order to apply velocity transformations in the future in an organized way
        # TODO: for now it doesn't do much that's useful, though, just permutes and unpermutes.
        # Pad the state vectors with ones
        padded_vectors = np.hstack([
            state_vectors,
            np.ones((state_vectors.shape[0], len(self.TRANSFORM_DIMS) - len(self.dims)))
        ]) if len(self.dims) < len(self.TRANSFORM_DIMS) else state_vectors
        ordered_states = padded_vectors @ self._permutation_matrix.T # TODO: stack these into one matrix multiplication in init
        ordered_states = ordered_states @ self._defaults_matrix.T # TODO: stack these into one matrix multiplication in init
        rotated_states = ordered_states @ self._rotation_matrix.T # TODO: stack these into one matrix multiplication in init
        # d_times = rotated_states[:, self.T_IDX:self.T_IDX+1] - self.defaults[Dim.T]
        # d_positions = d_times * rotated_states[:, self.V_IDX:self.V_IDX+3]
        translated_states = rotated_states
        #copies_of_origin = np.tile(self.origin, (state_vectors.shape[0], 1))
        # positions are self.POS_IDX:self.POS_IDX+3
        translated_states[:, self.POS_IDX:self.POS_IDX+3] += self.origin # + d_positions
        reordered_states = translated_states @ self._permutation_matrix # .T.T = no transpose; inv(permutation) = transpose(permutation) and double transpose cancels
        truncated_states = reordered_states[:, :len(self.dims)]
        return truncated_states
