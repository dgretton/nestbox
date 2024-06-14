from nestbox.proto_generated.twig_pb2 import Twig as GeneratedTwig
import numpy as np

def mean(sample):
    return np.array(sample.mean)

def covariance(measurement):
    n = len(measurement.mean)
    cov = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i, n):
            cov[i, j] = measurement.covariance.upper_triangle[idx]
            cov[j, i] = measurement.covariance.upper_triangle[idx]
            idx += 1
    return cov

class Twig:
    def __init__(self, bytes):
        self.twig = GeneratedTwig()
        self.twig.ParseFromString(bytes)
        if not self.twig.measurements or not self.twig.measurements[0].samples:
            return
        print(self.twig.measurements[0].samples[0])
        print(self.twig.measurements[0].samples[0].mean)
        print(self.twig.measurements[0].samples[0].covariance)
        print(self.twig.measurements[0].samples[0].covariance.upper_triangle)
        print(self.twig.measurements[0].transform)

    def all_means(self):
        return np.array([mean(sample) for measurement in self.twig.measurements for sample in measurement.samples])
    
    def covariance(self):
        return np.array([covariance(measurement) for measurement in self.twig.measurements])

    def __str__(self):
        return str(self.twig)
    
    def __repr__(self):
        return str(self.twig)