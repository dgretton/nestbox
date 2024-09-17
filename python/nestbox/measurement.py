from nestbox.feature import FeatureKey
from nestbox.protos import Dim
import numpy as np

class MeasurementType:
    NORMAL = 'NormalMeasurement'
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
