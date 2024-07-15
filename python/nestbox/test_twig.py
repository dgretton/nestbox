from nestbox.protos import Twig
from nestbox.proto_generated import twig_pb2
import numpy as np

def create_sample_twig():
    # Create a Twig message
    twig = twig_pb2.Twig()
    twig.coordSysId = "test_coord_sys"
    twig.streamId = "test_stream"

    # Create a MeasurementSet
    measurement_set = twig.measurements.add()
    measurement_set.dimensions.extend([twig_pb2.Dimension.X, twig_pb2.Dimension.Y, twig_pb2.Dimension.Z, twig_pb2.Dimension.T])
    
    # Add sample data
    sample = measurement_set.samples.add()
    sample.mean.extend([1.0, 2.0, 3.0, 4.0])
    sample.covariance.upper_triangle.extend([0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.4])

    # Set the isHomogeneous vector
    measurement_set.isHomogeneous.extend([True, True, True, False])  # Assume all but time are homogeneous

    # Set transformation matrix (identity for simplicity)
    measurement_set.transform.data.extend([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    return twig

def test_twig_serialization():
    # Create and serialize a Twig
    original_twig = create_sample_twig()
    serialized_data = original_twig.SerializeToString()

    # Deserialize the Twig
    deserialized_twig = twig_pb2.Twig()
    deserialized_twig.ParseFromString(serialized_data)

    # Create a Twig object using your custom class
    custom_twig = Twig(serialized_data)

    # Verify the deserialization
    assert custom_twig.coord_sys_id == original_twig.coordSysId
    assert custom_twig.stream_id == original_twig.streamId
    
    assert len(custom_twig.measurement_sets) == 1
    original_ms = original_twig.measurements[0]
    custom_ms = custom_twig.measurement_sets[0]

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

    print("All assertions passed. Twig serialization and deserialization working correctly.")

if __name__ == "__main__":
    test_twig_serialization()

