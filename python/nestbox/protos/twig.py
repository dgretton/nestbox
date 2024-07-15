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


class MeasurementSet:
    def __init__(self, twig_ms):
        self.dimensions = twig_ms.dimensions
        self.is_homogeneous = twig_ms.isHomogeneous
        self.means = []
        self.covariances = []
        for sample in twig_ms.samples:
            self.means.append(mean(sample))
            self.covariances.append(covariance(sample))
        self.means = np.stack(self.means)
        self.covariances = np.stack(self.covariances)
        self.transform = np.array(twig_ms.transform.data).reshape(3, 3)

    def __str__(self):
        return f"""MeasurementSet:
        dimensions: {self.dimensions}
        is_homogeneous: {self.is_homogeneous}
        means: {self.means}
        covariances: {self.covariances}
        transform: {self.transform}"""

    def __repr__(self):
        return self.__str__()


class Twig:
    def __init__(self, bytes):
        self.measurement_sets = []
        self.gentwig = GeneratedTwig()
        self.gentwig.ParseFromString(bytes)
        self.stream_id = self.gentwig.streamId
        self.coord_sys_id = self.gentwig.coordSysId
        if not self.gentwig.measurements or not self.gentwig.measurements[0].samples:
            return
        print(self.gentwig.measurements[0].samples[0])
        print(self.gentwig.measurements[0].samples[0].mean)
        print(self.gentwig.measurements[0].samples[0].covariance)
        print(self.gentwig.measurements[0].samples[0].covariance.upper_triangle)
        print(self.gentwig.measurements[0].transform)
        for ms in self.gentwig.measurements:
            self.measurement_sets.append(MeasurementSet(ms))
        print(self.measurement_sets[0])

    def all_means(self):
        return np.array([mean(sample) for measurement in self.gentwig.measurements for sample in measurement.samples])
    
    def covariance(self):
        return np.array([covariance(measurement) for measurement in self.gentwig.measurements])
    
    def __str__(self):
        return str(self.gentwig)
    
    def __repr__(self):
        return str(self.gentwig)