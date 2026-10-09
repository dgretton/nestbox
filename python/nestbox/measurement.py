from nestbox.feature import FeatureKey
from nestbox.protos import Dim
import numpy as np

class MeasurementType:
    NORMAL = 'NormalMeasurement'
    BEARING = 'BearingMeasurement'
    OPTIONS = 'OptionsMeasurement'
    COLLECTION = 'CollectionMeasurement'
    ORDERED = 'OrderedMeasurement'

def measurement(feature, mean, covariance):
    return NormalMeasurement(feature, mean, covariance)

# measurement class to be extended.
# Measurements have a feature, a mean vector and a covariance matrix at a minimum.
# you can get_sample() a measurement to get a (mean vector, covariance) tuple of numpy arrays. # TODO no probably not
class Measurement:
    def __init__(self, feature, dimensions=None, is_homogeneous=None, clear_key=None):
        if not isinstance(feature, FeatureKey):
            raise ValueError(f"Measurement must be initialized with a FeatureKey object. Got {feature}")
        self.feature = feature
        if dimensions is None:
            self.dimensions = [Dim.X, Dim.Y, Dim.Z]
        else:
            if not all (dim in Dim.all for dim in dimensions):
                raise ValueError(f"Measurement dimensions must be one of {', '.join(list(Dim.all))}. Got {dimensions}")
            self.dimensions = dimensions
        if is_homogeneous is None:
            self.is_homogeneous = [False] * len(self.dimensions)
        else:
            if not isinstance(is_homogeneous, list) or not all(isinstance(homog, bool) for homog in is_homogeneous):
                raise ValueError(f"Measurement is_homogeneous must be a list of booleans. Got {is_homogeneous}")
            self.is_homogeneous = is_homogeneous
        if clear_key is not None and not isinstance(clear_key, str):
            raise ValueError(f"Measurement clear_key must be a string. Got {clear_key}")
        self.clear_key = clear_key # this is a string that can be used to clear the measurement from the aligner, possibly along with a group of other measurements, to properly invalidate old measurements when new ones arrive

    def get_sample(self): # TODO this is a bad idea for measurements to have to implement this in general
        pass

    def __str__(self):
        return f"""{self.__class__.__name__}:
        feature: {self.feature}"""

    def __repr__(self):
        return self.__str__()
    

class NormalMeasurement(Measurement):
    def __init__(self, feature, mean, covariance, dimensions=None, is_homogeneous=None, clear_key=None):
        super().__init__(feature, dimensions, is_homogeneous, clear_key)
        self.mean = mean
        self.covariance = covariance

    def get_sample(self):
        return self.mean, self.covariance

    def __str__(self):
        # use parent class __str__ method and add mean and covariance
        return super().__str__() + f"""
        mean: {self.mean}
        covariance: {self.covariance}
        dimensions: ({', '.join(Dim.names(self.dimensions))})"""


class BearingMeasurement(Measurement):
    """A forward ray with tangent-plane angular noise (radians²), and optional log-range prior.

    A scalar angular_covariance is a standard deviation in radians. Matrix values are
    covariances in the supplied orthonormal tangent basis, not in azimuth/elevation.
    depth_prior is (positive range in meters, positive log-range standard deviation).
    """
    def __init__(self, feature, direction, angular_covariance, basis=None, origin=None,
                 depth_prior=None, dimensions=None, is_homogeneous=None, clear_key=None):
        super().__init__(feature, [Dim.I, Dim.J, Dim.K] if dimensions is None else dimensions,
                         is_homogeneous, clear_key)
        self.direction = np.asarray(direction, dtype=float).copy()
        if self.direction.shape != (3,) or not np.all(np.isfinite(self.direction)) or np.linalg.norm(self.direction) == 0:
            raise ValueError('direction must be a finite nonzero 3-vector')
        self.direction /= np.linalg.norm(self.direction)
        self.basis = self.tangent_basis(self.direction) if basis is None else np.asarray(basis, dtype=float).copy()
        if (self.basis.shape != (3, 2) or not np.all(np.isfinite(self.basis))
                or not np.allclose(self.basis.T @ self.basis, np.eye(2), atol=1e-8)
                or not np.allclose(self.basis.T @ self.direction, 0, atol=1e-8)):
            raise ValueError('basis must have two orthonormal columns perpendicular to direction')
        cov = np.asarray(angular_covariance, dtype=float)
        if cov.ndim == 0:
            if not np.isfinite(cov) or cov <= 0:
                raise ValueError('angular sigma must be positive and finite')
            cov = np.eye(2) * float(cov) ** 2
        if (cov.shape != (2, 2) or not np.all(np.isfinite(cov))
                or not np.allclose(cov, cov.T, rtol=1e-8, atol=1e-15)
                or np.min(np.linalg.eigvalsh(cov)) <= 0):
            raise ValueError('angular_covariance must be a positive definite symmetric 2x2 matrix')
        self.angular_covariance = cov.copy()
        self.origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float).copy()
        if self.origin.shape != (3,) or not np.all(np.isfinite(self.origin)):
            raise ValueError('origin must be a finite 3-vector')
        self.depth_prior = None
        if depth_prior is not None:
            prior = np.asarray(depth_prior, dtype=float)
            if prior.shape != (2,) or not np.all(np.isfinite(prior)) or np.any(prior <= 0):
                raise ValueError('depth_prior must be (positive rho0, positive sigma_log_rho)')
            self.depth_prior = tuple(prior)

    @staticmethod
    def tangent_basis(direction):
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        e = np.eye(3)[np.argmin(np.abs(d))]
        e = e - d * (d @ e)
        e /= np.linalg.norm(e)
        return np.column_stack((e, np.cross(d, e)))

    @classmethod
    def from_unit_vector(cls, feature, direction, covariance, **kwargs):
        """Decode a Twig [I,J,K] sample; its 3x3 covariance's tangent block is angular."""
        # Validate direction before constructing the basis.
        m = cls(feature, direction, 1.0, **kwargs)
        cov = np.asarray(covariance, dtype=float)
        if cov.shape != (3, 3) or not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T):
            raise ValueError('unit-vector covariance must be a finite symmetric 3x3 matrix')
        return cls(feature, m.direction, m.basis.T @ cov @ m.basis, **dict(kwargs, basis=m.basis))

    def to_dict(self):
        return dict(type=MeasurementType.BEARING, feature=str(self.feature), direction=self.direction.tolist(),
                    angular_covariance=self.angular_covariance.tolist(), basis=self.basis.tolist(),
                    origin=self.origin.tolist(), depth_prior=self.depth_prior, dimensions=self.dimensions,
                    is_homogeneous=self.is_homogeneous, clear_key=self.clear_key)


def measurement_from_dict(data):
    from nestbox.feature import to_feature
    common = dict(dimensions=data.get('dimensions'), is_homogeneous=data.get('is_homogeneous'),
                  clear_key=data.get('clear_key'))
    feature = to_feature(data['feature'])
    if data['type'] == MeasurementType.NORMAL:
        return NormalMeasurement(feature, data['mean'], data['covariance'], **common)
    if data['type'] == MeasurementType.BEARING:
        return BearingMeasurement(feature, data['direction'], data['angular_covariance'],
                                  basis=data.get('basis'), origin=data.get('origin'),
                                  depth_prior=data.get('depth_prior'), **common)
    raise ValueError(f"Unsupported measurement type: {data['type']}")


# TODO ok this is totally unhinged.
# class OptionsMeasurement(Measurement):
#     def __init__(self, feature, options, weights=None):
#         if not all(isinstance(opt, Measurement) for opt in options):
#             raise ValueError(f"OptionsMeasurement must be initialized with a list of Measurement objects. Got {options}")
#         super().__init__(feature)
#         if weights is None:
#             weights = np.ones(len(options))
#         self.options = options

#     def get_sample(self, weights=None):
#         return np.random.choice(self.options, p=(self.weights if weights is None else weights)).get_sample()

#     def __str__(self):
#         # use parent class __str__ method and add options
#         return super().__str__() + f"""
#         options: {self.options}"""
