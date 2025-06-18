import json

import matplotlib.pyplot as plt
import numpy as np

from manifoldmap import ProcrustesBasedPoseExtractor

with open("measurement_data_reflected.json", "r") as f:
    data = json.load(f)
single_hand = np.array(data["measurements"]["3_4_1"]["vr_tracking_points"])
print(single_hand.shape)

with open("angled_hand_mount_hand_positions.json", "r") as f:
    data = json.load(f)
reference_hand = np.array(data["hand_points"])
reference_hand /= 1000.0 # convert mm to m
print(reference_hand.shape)

# Calculate the pose
pose_extractor = ProcrustesBasedPoseExtractor(reference_hand)
pose = pose_extractor.extract_pose(single_hand)
transformed_hand = pose.transform_points(reference_hand)

# Visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot both datasets
ax.scatter(single_hand[:, 0], single_hand[:, 1], single_hand[:, 2], c='blue', label='Source', alpha=0.6)
ax.scatter(reference_hand[:, 0], reference_hand[:, 1], reference_hand[:, 2], c='red', label='Reference', alpha=0.6)
ax.scatter(transformed_hand[:, 0], transformed_hand[:, 1], transformed_hand[:, 2], c='green', label='Transformed', alpha=0.6)

# Calculate the overall data range and center
all_data = np.vstack([single_hand, reference_hand])
ranges = [
    all_data[:, 0].max() - all_data[:, 0].min(),
    all_data[:, 1].max() - all_data[:, 1].min(),
    all_data[:, 2].max() - all_data[:, 2].min()
]
max_range = max(ranges)
centers = [
    (all_data[:, 0].max() + all_data[:, 0].min()) / 2,
    (all_data[:, 1].max() + all_data[:, 1].min()) / 2,
    (all_data[:, 2].max() + all_data[:, 2].min()) / 2
]

# Set equal axis limits using the largest range
margin = max_range * 0.1  # 10% margin
half_range = max_range / 2 + margin

ax.set_xlim(centers[0] - half_range, centers[0] + half_range)
ax.set_ylim(centers[1] - half_range, centers[1] + half_range)
ax.set_zlim(centers[2] - half_range, centers[2] + half_range)

# Now set_box_aspect will work correctly
ax.set_box_aspect([1,1,1])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Manifold Point Mapping Visualization')
ax.legend()

plt.tight_layout()
plt.show()








