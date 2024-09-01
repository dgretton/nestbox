import nestbox
import numpy as np
from nestbox import Dim

from nestbox.test_live_data import hand_points

# Create coordinate systems
nestbox.create_coordinate_system('cs1')
nestbox.create_coordinate_system('cs2')
nestbox.create_coordinate_system('base')

# Function to create measurements for a hand
def create_hand_measurements(stream_id, mirror=False):
    measurements = []
    for i, point in enumerate(hand_points):
        point = np.array(point)
        if mirror:
            point[0] *= -1
        measurements.append({
            "type": "NormalMeasurement",
            "feature": f"{stream_id}_{i}",
            "mean": point.tolist(),
            "covariance": [[0.0001, 0, 0], [0, 0.0001, 0], [0, 0, 0.0001]],  # Small covariance for demonstration
            "dimensions": [Dim.X, Dim.Y, Dim.Z],
            "is_homogeneous": [False, False, False]
        })
    return measurements

# Add measurements to cs1 (left hand, mirrored)
left_hand_measurements = create_hand_measurements('lefthand', mirror=True)
nestbox.add_measurements('cs1', left_hand_measurements)

# Add measurements to cs2 (right hand)
right_hand_measurements = create_hand_measurements('righthand')
nestbox.add_measurements('cs2', right_hand_measurements)

# Start the alignment process
nestbox.start_alignment()

print("Test setup complete. The aligner is now running with live data.")
print("You can now run test_toplevel_api.py to interact with the system.")
