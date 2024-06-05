from nestbox.proto_generated.twig_pb2 import Twig as ProtoGenTwig
import numpy as np

def mean(measurement):
    return np.array([measurement.mean])

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
        self.twig = ProtoGenTwig()
        self.twig.ParseFromString(bytes)
        print(self.twig.measurements[0])
        print(self.twig.measurements[0].mean)
        print(self.twig.measurements[0].covariance)
        print(self.twig.measurements[0].covariance.upper_triangle)
        print(self.twig.measurements[0].transform)

    def means(self):
        return np.array([mean(measurement) for measurement in self.twig.measurements])
    
    def covariance(self):
        return np.array([covariance(measurement) for measurement in self.twig.measurements])

    def __str__(self):
        return str(self.twig)
    
    def __repr__(self):
        return str(self.twig)