from nestbox.protos import Twig, MeasurementSet
from nestbox.proto_generated import twig_pb2
from nestbox.numutil import covariance_to_upper_triangle
import numpy as np

class SampleData:
    coordSysId = "test_coord_sys"
    streamId = "test_stream"
    dimensions = [twig_pb2.Dimension.X, twig_pb2.Dimension.Y, twig_pb2.Dimension.Z, twig_pb2.Dimension.T]
    mean = [1.0, 2.0, 3.0, 4.0]
    covariance = [[0.1, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0], [0.0, 0.0, 0.3, 0.0], [0.0, 0.0, 0.0, 0.4]]
    isHomogeneous = [True, True, True, False]
    transform = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def create_sample_raw_twig():
    # Create a Twig message
    twig_raw = twig_pb2.Twig()
    twig_raw.coordSysId = "test_coord_sys"
    twig_raw.streamId = "test_stream"

    # Create a MeasurementSet
    measurement_set = twig_raw.measurements.add()
    #measurement_set.dimensions.extend([twig_pb2.Dimension.X, twig_pb2.Dimension.Y, twig_pb2.Dimension.Z, twig_pb2.Dimension.T])
    measurement_set.dimensions.extend(SampleData.dimensions)
    
    # Add sample data
    sample = measurement_set.samples.add()
    # sample.mean.extend([1.0, 2.0, 3.0, 4.0])
    # sample.covariance.upper_triangle.extend([0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.4])
    sample.mean.extend(SampleData.mean)
    sample.covariance.upper_triangle.extend(covariance_to_upper_triangle(SampleData.covariance))

    # Set the isHomogeneous vector
    # measurement_set.isHomogeneous.extend([True, True, True, False])  # Assume all but time are homogeneous
    measurement_set.isHomogeneous.extend(SampleData.isHomogeneous)

    # Set transformation matrix (identity for simplicity)
    # measurement_set.transform.data.extend([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    measurement_set.transform.data.extend(SampleData.transform)

    return twig_raw

def create_sample_twig():
    ms = MeasurementSet(samples=[(SampleData.mean, SampleData.covariance)],
                        dimensions=SampleData.dimensions,
                        is_homogeneous=SampleData.isHomogeneous,
                        transform=SampleData.transform)
    return Twig(coord_sys_id=SampleData.coordSysId,
                stream_id=SampleData.streamId,
                measurement_sets=[ms])

def test_twig_serialization():
    # Create and serialize Twigs
    original_raw_twig = create_sample_raw_twig()
    serialized_raw_twig = original_raw_twig.SerializeToString()
    original_custom_twig = create_sample_twig()
    json_serialized_custom_twig = original_custom_twig.to_json_str()
    bytes_serialized_custom_twig = original_custom_twig.to_bytes()

    # Deserialize the Twigs
    deserialized_raw_twig = twig_pb2.Twig()
    deserialized_raw_twig.ParseFromString(serialized_raw_twig)

    # Create a Twig object using custom class
    custom_twig_from_raw = Twig().load_bytes(serialized_raw_twig)
    custom_twig_from_custom_json = Twig().load_json_str(json_serialized_custom_twig)
    custom_twig_from_custom_bytes = Twig().load_bytes(bytes_serialized_custom_twig)

    # Verify the deserialization
    assert (
        original_raw_twig.coordSysId
        == deserialized_raw_twig.coordSysId
        == custom_twig_from_raw.coord_sys_id
        == custom_twig_from_custom_json.coord_sys_id
        == custom_twig_from_custom_bytes.coord_sys_id
    )
    # assert custom_twig_from_raw.coord_sys_id == original_raw_twig.coordSysId == deserialized_raw_twig.coordSysId == custom_twig_from_custom_json.coord_sys_id == custom_twig_from_custom_bytes.coord_sys_id
    assert (
        original_raw_twig.streamId
        == deserialized_raw_twig.streamId
        == custom_twig_from_raw.stream_id
        == custom_twig_from_custom_json.stream_id
        == custom_twig_from_custom_bytes.stream_id
    )
    # assert custom_twig_from_raw.stream_id == original_raw_twig.streamId == deserialized_raw_twig.streamId == custom_twig_from_custom_json.stream_id
    
    assert (
        len(original_raw_twig.measurements)
        == len(deserialized_raw_twig.measurements)
        == len(custom_twig_from_raw.measurement_sets)
        == len(custom_twig_from_custom_json.measurement_sets)
        == len(custom_twig_from_custom_bytes.measurement_sets)
        == 1  # Only one measurement set in the custom Twig
    )
    # assert len(custom_twig_from_raw.measurement_sets) == 1
    # assert len(custom_twig_from_custom_json.measurement_sets) == 1
    original_ms = original_raw_twig.measurements[0]

    def validate_custom_ms(custom_ms):
        custom_ms = custom_twig_from_raw.measurement_sets[0]

        # Debug prints
        print("Type of custom_ms.transform:", type(custom_ms.transform))
        print("Shape of custom_ms.transform:", custom_ms.transform.shape if hasattr(custom_ms.transform, 'shape') else "No shape attribute")
        print("Content of custom_ms.transform:", custom_ms.transform)

        original_transform = np.array(original_ms.transform.data).reshape(3, 3)
        print("Type of original transform:", type(original_transform))
        print("Shape of original transform:", original_transform.shape)
        print("Content of original transform:", original_transform)

        # Try to convert custom_ms.transform to numpy array if it's not already
        if not isinstance(custom_ms.transform, np.ndarray):
            custom_ms.transform = np.array(custom_ms.transform)

        # Now try the comparison
        try:
            assert np.allclose(custom_ms.transform, original_transform)
            print("Transforms are close.")
        except Exception as e:
            print("Error in comparing transforms:", str(e))

        assert list(custom_ms.dimensions) == list(original_ms.dimensions)
        assert np.allclose(custom_ms.means[0], original_ms.samples[0].mean)
        assert np.allclose(custom_ms.covariances[0], np.array([
            [0.1, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.4]
        ]))
        assert list(custom_ms.is_homogeneous) == list(original_ms.isHomogeneous)
        assert np.allclose(custom_ms.transform, np.array(original_ms.transform.data).reshape(3, 3))

    custom_ms_from_raw = custom_twig_from_raw.measurement_sets[0]
    custom_ms_from_custom_json = custom_twig_from_custom_json.measurement_sets[0]
    custom_ms_from_custom_bytes = custom_twig_from_custom_bytes.measurement_sets[0]
    validate_custom_ms(custom_ms_from_raw)
    validate_custom_ms(custom_ms_from_custom_json)
    validate_custom_ms(custom_ms_from_custom_bytes)

    print("All assertions passed. Twig serialization and deserialization working correctly.")

if __name__ == "__main__":
    test_twig_serialization()

