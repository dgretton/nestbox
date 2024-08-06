import nestbox
from nestbox.proto_generated.twig_pb2 import Twig as GeneratedTwig
from nestbox.numutil import upper_triangle_to_covariance, covariance_to_upper_triangle, coerce_numpy
import numpy as np
import json

def get_proto_mean(sample):
    return coerce_numpy(sample.mean)

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
                # mean should be 1d, cov should be 2d, dim should match
                mean = coerce_numpy(mean)
                cov = coerce_numpy(cov)
                if len(mean.shape) != 1:
                    raise ValueError(f"Mean must be 1-dimensional. Got {mean}")
                if len(cov.shape) != 2:
                    raise ValueError(f"Covariance must be 2-dimensional. Got {cov}")
                if cov.shape[0] != cov.shape[1] or cov.shape[0] != len(mean):
                    raise ValueError(f"Covariance must be square and have the same number of rows as the mean vector ({len(mean)}). Got {cov}")
                self.means.append(coerce_numpy(mean))
                self.covariances.append(coerce_numpy(cov))
        self.means = np.stack(self.means) if self.means else np.array([[]])
        self.covariances = np.stack(self.covariances) if self.covariances else np.array([[[]]])
        dimensions = list(dimensions) if dimensions else None
        is_homogeneous = list(is_homogeneous) if is_homogeneous else None
        self._validate_dims(dimensions)
        self._validate_homogeneous(is_homogeneous)
        self.dimensions = dimensions
        self.is_homogeneous = is_homogeneous
        self.transform = coerce_numpy(transform)

    @staticmethod
    def from_twig_ms(twig_ms):
        samples = []
        for sample in twig_ms.samples:
            samples.append((get_proto_mean(sample), get_proto_covariance(sample)))
        meas_set = MeasurementSet(samples=samples)
        dims = list(twig_ms.dimensions) if twig_ms.dimensions else None
        MeasurementSet._validate_dims(dims)
        meas_set.dimensions = dims
        homog = list(twig_ms.isHomogeneous) if twig_ms.isHomogeneous else None
        MeasurementSet._validate_homogeneous(homog)
        meas_set.is_homogeneous = homog
        meas_set.means = np.stack(meas_set.means)
        meas_set.covariances = np.stack(meas_set.covariances)
        meas_set.transform = coerce_numpy(twig_ms.transform.data).reshape(3, 3)
        return meas_set

    def to_json(self):
        # sample:
        # {
        #     "dimensions": [0, 1, 2],
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
            "dimensions": self.dimensions if self.dimensions and MeasurementSet._validate_dims(self.dimensions) else [Dim.X, Dim.Y, Dim.Z],
            "is_homogeneous": self.is_homogeneous if self.is_homogeneous and MeasurementSet._validate_homogeneous(self.is_homogeneous) else [False] * len(self.dimensions),
            "samples": [
                {
                    "mean": mean.tolist(),
                    "covariance": {
                        "upper_triangle": covariance_to_upper_triangle(cov).tolist()
                    }
                } for mean, cov in zip(self.means, self.covariances)
            ],
            "transform": {
                "data": self.transform.tolist() if self.transform is not None else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            }
        }
    
    def to_json_str(self):
        return json.dumps(self.to_json())
    
    def load_json(self, json_dict):
        if not isinstance(json_dict, dict):
            raise ValueError(f"MeasurementSet must be initialized with a dictionary. Got {type(json_dict)}")
        self.dimensions = json_dict.get("dimensions", [Dim.X, Dim.Y, Dim.Z])
        self._validate_dims(self.dimensions)
        self.is_homogeneous = json_dict.get("is_homogeneous", [False] * len(self.dimensions))
        self._validate_homogeneous(self.is_homogeneous)
        self.means = []
        self.covariances = []
        for sample in json_dict.get("samples", []):
            mean = coerce_numpy(sample["mean"])
            cov = upper_triangle_to_covariance(coerce_numpy(sample["covariance"]["upper_triangle"]))
            self.means.append(mean)
            self.covariances.append(cov)
        self.transform = coerce_numpy(json_dict.get("transform", {}).get("data", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])).reshape(3, 3)
        return self
    
    def load_json_str(self, json_str):
        if not isinstance(json_str, str):
            raise ValueError(f"MeasurementSet must be initialized with a string. Got {json_str}")
        return self.load_json(json.loads(json_str))

    @staticmethod
    def _validate_dims(dims):
        assert dims is None or isinstance(dims, list), f"MeasurementSet dimensions must be a list. Got type {type(dims)}"
        if dims is not None and not all(dim in Dim.all for dim in dims):
            print('MeasurementSet dimensions must be one of {', ', '.join(Dim.all), '}. Got {', dims, '}')
            raise ValueError(f"MeasurementSet dimensions must be one of {', '.join(Dim.all)}. Got {dims}")

    @staticmethod
    def _validate_homogeneous(is_homogeneous):
        assert is_homogeneous is None or isinstance(is_homogeneous, list), f"MeasurementSet is_homogeneous must be a list. Got type {type(is_homogeneous)}"
        if is_homogeneous is not None and not all(isinstance(homogeneous, bool) for homogeneous in is_homogeneous):
            raise ValueError(f"MeasurementSet is_homogeneous must be a list of booleans. Got {is_homogeneous}")

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
    def __init__(self, stream_id=None, coord_sys_id=None, measurement_sets=None):
        self.stream_id = stream_id
        self.coord_sys_id = coord_sys_id
        if not measurement_sets:
            self.measurement_sets = []
        else:
            if not all(isinstance(ms, MeasurementSet) for ms in measurement_sets):
                raise ValueError(f"Twig must be initialized with a list of MeasurementSet objects. Got {measurement_sets}")
            self.measurement_sets = measurement_sets
        self.gentwig = None

    def load_bytes(self, twig_bytes):
        if not isinstance(twig_bytes, bytes):
            raise ValueError(f"Twig must be initialized with a bytes object. Got {twig_bytes}")
        self.measurement_sets = []
        self.gentwig = GeneratedTwig()
        self.gentwig.ParseFromString(twig_bytes)
        self.stream_id = self.gentwig.streamId
        self.coord_sys_id = self.gentwig.coordSysId
        for ms in self.gentwig.measurements:
            self.measurement_sets.append(MeasurementSet.from_twig_ms(ms))
        return self
    
    def to_bytes(self):
        #example of twig building
        # def create_sample_raw_twig():
        #     # Create a Twig message
        #     twig_raw = twig_pb2.Twig()
        #     twig_raw.coordSysId = "test_coord_sys"
        #     twig_raw.streamId = "test_stream"

        #     # Create a MeasurementSet
        #     measurement_set = twig_raw.measurements.add()
        #     #measurement_set.dimensions.extend([twig_pb2.Dimension.X, twig_pb2.Dimension.Y, twig_pb2.Dimension.Z, twig_pb2.Dimension.T])
        #     measurement_set.dimensions.extend(SampleData.dimensions)
            
        #     # Add sample data
        #     sample = measurement_set.samples.add()
        #     # sample.mean.extend([1.0, 2.0, 3.0, 4.0])
        #     # sample.covariance.upper_triangle.extend([0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.4])
        #     sample.mean.extend(SampleData.mean)
        #     sample.covariance.upper_triangle.extend(covariance_to_upper_triangle(SampleData.covariance))

        #     # Set the isHomogeneous vector
        #     # measurement_set.isHomogeneous.extend([True, True, True, False])  # Assume all but time are homogeneous
        #     measurement_set.isHomogeneous.extend(SampleData.isHomogeneous)

        #     # Set transformation matrix (identity for simplicity)
        #     # measurement_set.transform.data.extend([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        #     measurement_set.transform.data.extend(SampleData.transform)

        #     return twig_raw

        self.gentwig = GeneratedTwig()
        self.gentwig.streamId = self.stream_id
        self.gentwig.coordSysId = self.coord_sys_id
        print('got here')
        for ms in self.measurement_sets:
            print('got here 2')
            raw_ms = self.gentwig.measurements.add()
            print('got here 2.1')
            raw_ms.dimensions.extend(ms.dimensions)
            print('got here 2.2')
            raw_ms.isHomogeneous.extend(ms.is_homogeneous)
            print('got here 3')
            for mean, cov in zip(ms.means, ms.covariances):
                print('got here 4')
                sample = raw_ms.samples.add()
                sample.mean.extend(mean)
                sample.covariance.upper_triangle.extend(covariance_to_upper_triangle(cov))
            print('got here 5')
            raw_ms.transform.data.extend(ms.transform.flatten())
        print('got here 6')
        return self.gentwig.SerializeToString()

    def all_means(self):
        return coerce_numpy([get_proto_mean(sample) for measurement in self.gentwig.measurements for sample in measurement.samples])
    
    def covariance(self):
        return coerce_numpy([get_proto_covariance(measurement) for measurement in self.gentwig.measurements])
    
    def to_json(self):
        return {
            "stream_id": self.stream_id,
            "coord_sys_id": self.coord_sys_id,
            "measurement_sets": [ms.to_json() for ms in self.measurement_sets]
        }
    
    def to_json_str(self):
        return json.dumps(self.to_json())
    
    def load_json(self, json_dict):
        if not isinstance(json_dict, dict):
            raise ValueError(f"Twig must be initialized with a dictionary. Got {json_dict}")
        self.stream_id = json_dict.get("stream_id")
        self.coord_sys_id = json_dict.get("coord_sys_id")
        self.measurement_sets = [MeasurementSet().load_json(ms) for ms in json_dict.get("measurement_sets", [])]
        return self
    
    def load_json_str(self, json_str):
        if not isinstance(json_str, str):
            raise ValueError(f"Twig must be initialized with a string. Got {json_str}")
        return self.load_json(json.loads(json_str))
    
    def __str__(self):
        return str(self.gentwig)
    
    def __repr__(self):
        return str(self.gentwig)


class Dim:
    gd = nestbox.proto_generated.twig_pb2.Dimension
    X = gd.X
    Y = gd.Y
    Z = gd.Z
    T = gd.T
    VX = gd.VX
    VY = gd.VY
    VZ = gd.VZ
    I = gd.I
    J = gd.J
    K = gd.K

    def __init__(self) -> None:
        self.str_map = {
            'X': self.X,
            'Y': self.Y,
            'Z': self.Z,
            'T': self.T,
            'VX': self.VX,
            'VY': self.VY,
            'VZ': self.VZ,
            'I': self.I,
            'J': self.J,
            'K': self.K
        }
        self.all_dims = sorted(self.str_map.values())

    def __getitem__(self, key):
        return self.str_map[key]

    def name(self, dim: int):
        return self.gd.Name(dim)
    
    def names(self, dims):
        return [self.name(dim) for dim in dims]
    
    @property
    def all(self):
        return self.all_dims[:]

Dim = Dim()

if __name__ == "__main__":
    # test dim class
    print('Dim class test:')
    print(f'Dim.X: {Dim.X}')
    print(f'Dim.VX: {Dim.VX}')
    print(f'Dim["VX"]: {Dim["VX"]}')
    print(f'Dim.name(Dim.K): {Dim.name(Dim.K)}')
