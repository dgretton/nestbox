from nestbox.proto_generated.twig_pb2 import Twig as GeneratedTwig
from nestbox.numutil import upper_triangle_to_covariance, coerce_numpy
import numpy as np
import json

def get_proto_mean(sample):
    return np.array(sample.mean)

def get_proto_covariance(measurement):
    # TODO delete me when we know it's a good replacement
    # n = len(measurement.mean)
    # cov = np.zeros((n, n))
    # idx = 0
    # for i in range(n):
    #     for j in range(i, n):
    #         cov[i, j] = measurement.covariance.upper_triangle[idx]
    #         cov[j, i] = measurement.covariance.upper_triangle[idx]
    #         idx += 1
    # return cov
    upper_triangle = coerce_numpy(measurement.covariance.upper_triangle)
    return upper_triangle_to_covariance(upper_triangle)


class MeasurementSet:
    def __init__(self, samples=None, dimensions=None, is_homogeneous=None, transform=None):
        self.means = []
        self.covariances = []
        if samples:
            for mean, cov in samples:
                self.means.append(np.array(mean))
                self.covariances.append(np.array(cov))
        self.means = np.stack(self.means)
        self.covariances = np.stack(self.covariances)
        self.dimensions = dimensions
        self.is_homogeneous = is_homogeneous
        self.transform = np.array(transform)

    @staticmethod
    def from_twig_ms(twig_ms):
        samples = []
        for sample in twig_ms.samples:
            samples.append((get_proto_mean(sample), get_proto_covariance(sample)))
        meas_set = MeasurementSet(samples=samples)
        meas_set.dimensions = twig_ms.dimensions
        meas_set.is_homogeneous = twig_ms.isHomogeneous
        meas_set.means = np.stack(meas_set.means)
        meas_set.covariances = np.stack(meas_set.covariances)
        meas_set.transform = np.array(twig_ms.transform.data).reshape(3, 3)

    def to_json(self):
        # sample:
        # {
        #     "dimensions": ["X", "Y", "Z"],
        #     "is_homogeneous": [false, false, false],
        #     "samples": [
        #         {
        #             "mean": [1.0, 2.0, 3.0],
        #             "covariance": {
        #                 "upper_triangle": [0.01, 0.002, 0.03, 0.003, 0.004, 0.05]
        #             }
        #         },
        #         {
        #             "mean": [1.5, 2.5, 3.5],
        #             "covariance": {
        #                 "upper_triangle": [0.02, 0.003, 0.04, 0.004, 0.005, 0.06]
        #             }
        #         }
        #     ],
        #     "transform": {
        #         "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        #     }
        # }
        return {
            "dimensions": self.dimensions if self.dimensions else ["X", "Y", "Z"],
            "is_homogeneous": self.is_homogeneous if self.is_homogeneous else [False] * len(self.dimensions),
            "samples": [
                {
                    "mean": mean.tolist(),
                    "covariance": {
                        "matrix": cov.tolist()
                    }
                } for mean, cov in zip(self.means, self.covariances)
            ],
            "transform": {
                "data": self.transform.tolist() if self.transform else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            }
        }
    
    def to_json_str(self):
        return json.dumps(self.to_json())

    def __str__(self):
        return f"""MeasurementSet:
        dimensions: {self.dimensions}
        is_homogeneous: {self.is_homogeneous}
        means: {self.means}
        covariances: {self.covariances}
        transform: {self.transform}"""

    def __repr__(self):
        return self.to_json_str()


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
        return np.array([get_proto_mean(sample) for measurement in self.gentwig.measurements for sample in measurement.samples])
    
    def covariance(self):
        return np.array([get_proto_covariance(measurement) for measurement in self.gentwig.measurements])
    
    def __str__(self):
        return str(self.gentwig)
    
    def __repr__(self):
        return str(self.gentwig)