from nestbox.protos import Twig, MeasurementSet, Dim
from nestbox.measurement import MeasurementType, NormalMeasurement
from nestbox.feature import StrFeatureKey

def sample_twig():
    import numpy as np
    # Create a Twig instance
    twig = Twig(stream_id="test_stream", coord_sys_id="test_coord_sys")

    # Define common covariance
    cov = np.eye(3) * 0.02

    # Create the first MeasurementSet
    ms1 = MeasurementSet(
        samples=[
            (np.array([0, 0, 1]), cov),
            (np.array([0, 1, 0]), cov),
            (np.array([-1, 0, 0]), cov)
        ],
        dimensions=[Dim.I, Dim.J, Dim.K],
        is_homogeneous=[False, False, False],
        transform=np.eye(3)
    )

    # Create the second MeasurementSet
    ms2 = MeasurementSet(
        samples=[
            (np.array([0, 0, -1]), cov),
            (np.array([0, 1, 0]), cov),
            (np.array([1, 0, 0]), cov)
        ],
        dimensions=[Dim.I, Dim.J, Dim.K],
        is_homogeneous=[False, False, False],
        transform=np.eye(3)
    )

    # Add MeasurementSets to the Twig
    twig.measurement_sets = [ms1, ms2]

    return twig


class SampleRouter:
    def __init__(self, routing_info):
        self.routing_info = routing_info

    def unpack_measurements(self, twig):
        # so you have a twig. It has a bunch of measurement sets, each full of samples.
        # you've been configured with the parameters needed to map indices like meas_sets[2][0] to complex Measurement objects
        # each measurement needs to have a Strfeaturekey, based on a feature URI
        # example twig input, containing 2 measurement sets, each with 6 samples defined over IJK dimensions:
        # (these represent orthonormal basis vectors, so they are to be interpreted 3 at a time: I-hat, J-hat, K-hat)
        measurements = []
        
        # Verify that the incoming twig matches the expected stream_id
        if twig.stream_id != self.routing_info["stream_id"]:
            raise ValueError(f"Twig stream ID '{twig.stream_id}' does not match expected stream ID '{self.routing_info['stream_id']}'")

        for measurement_info in self.routing_info["measurements"]:
            if measurement_info["type"] == "normal":
                feature_key = StrFeatureKey(measurement_info["feature_uri"])
                clear_key = measurement_info["clear_key"]
                set_index = measurement_info["sample_pointer"]["set"]
                sample_index = measurement_info["sample_pointer"]["sample"]

                # Ensure the measurement set and sample exist
                if set_index >= len(twig.measurement_sets):
                    raise IndexError(f"Measurement set {set_index} does not exist in twig")
                if sample_index >= len(twig.measurement_sets[set_index].means):
                    raise IndexError(f"Sample {sample_index} does not exist in measurement set {set_index}")

                measurement_set = twig.measurement_sets[set_index]
                mean = measurement_set.means[sample_index]
                cov = measurement_set.covariances[sample_index]
                dims = measurement_set.dimensions
                measurement = NormalMeasurement(feature_key, mean, cov, dimensions=dims, clear_key=clear_key)
                measurements.append(measurement)
            else:
                raise ValueError(f"Unsupported measurement type: {measurement_info['type']}")

        return measurements

def test_sample_router():
    # Create a sample Twig
    twig = sample_twig()

    # Create a sample routing configuration
    routing_info = {
        "stream_id": "test_stream",
        "measurements": [
            {
                "type": "normal",
                "feature_uri": "nestbox:feature/tag/features/point/feature1/position",
                "sample_pointer": {
                    "set": 0,
                    "sample": 0
                },
                "clear_key": "nestbox:feature/tag/features/point"
            },
            {
                "type": "normal",
                "feature_uri": "nestbox:feature/tag/features/point/feature2/position",
                "sample_pointer": {
                    "set": 0,
                    "sample": 1
                },
                "clear_key": "nestbox:feature/tag/features/point"
            },
            {
                "type": "normal",
                "feature_uri": "nestbox:feature/tag/features/point/feature3/position",
                "sample_pointer": {
                    "set": 0,
                    "sample": 2
                },
                "clear_key": "nestbox:feature/tag/features/point"
            },
            {
                "type": "normal",
                "feature_uri": "nestbox:feature/tag/features/point/feature4/position",
                "sample_pointer": {
                    "set": 1,
                    "sample": 0
                },
                "clear_key": "nestbox:feature/tag/features/point"
            },
            {
                "type": "normal",
                "feature_uri": "nestbox:feature/tag/features/point/feature5/position",
                "sample_pointer": {
                    "set": 1,
                    "sample": 1
                },
                "clear_key": "nestbox:feature/tag/features/point"
            },
            {
                "type": "normal",
                "feature_uri": "nestbox:feature/tag/features/point/feature6/position",
                "sample_pointer": {
                    "set": 1,
                    "sample": 2
                },
                "clear_key": "nestbox:feature/tag/features/point"
            }
        ]
    }

    # Create a SampleRouter
    router = SampleRouter(routing_info)

    # Unpack the measurements
    measurements = router.unpack_measurements(twig)
    print(measurements)

if __name__ == "__main__":
    test_sample_router()
