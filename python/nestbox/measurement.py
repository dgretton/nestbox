from feature import FeatureKey
import numpy as np
# measurement class to be extended.
# Measurements have a feature, a mean vector and a covariance matrix at a minimum.
# you can get_sample() a measurement to get a (mean vector, covariance) tuple of numpy arrays.

def measurement(feature, mean, covariance):
    return NormalMeasurement(feature, mean, covariance)


class Measurement:
    def __init__(self, feature):
        if not isinstance(feature, FeatureKey):
            raise ValueError(f"Measurement must be initialized with a FeatureKey object. Got {feature}")
        self.feature = feature

    def get_sample(self):
        pass

    def __str__(self):
        return f"""{self.__class__.__name__}:
        feature: {self.feature}"""

    def __repr__(self):
        return self.__str__()
    

class NormalMeasurement(Measurement):
    def __init__(self, feature, mean, covariance):
        super().__init__(feature)
        self.mean = mean
        self.covariance = covariance

    def get_sample(self):
        return self.mean, self.covariance

    def __str__(self):
        # use parent class __str__ method and add mean and covariance
        return super().__str__() + f"""
        mean: {self.mean}
        covariance: {self.covariance}"""


class OptionsMeasurement(Measurement):
    def __init__(self, feature, options, weights=None):
        if not all(isinstance(opt, Measurement) for opt in options):
            raise ValueError(f"OptionsMeasurement must be initialized with a list of Measurement objects. Got {options}")
        super().__init__(feature)
        if weights is None:
            weights = np.ones(len(options))
        self.options = options

    def get_sample(self, weights=None):
        return np.random.choice(self.options, p=(self.weights if weights is None else weights)).get_sample()

    def __str__(self):
        # use parent class __str__ method and add options
        return super().__str__() + f"""
        options: {self.options}"""
