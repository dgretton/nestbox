import copy
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

from mpl_toolkits.mplot3d import Axes3D

from nestbox.metrology.manifoldmap import PolynomialMapper, LerpMapper

"""
Resample hand points to smooth out noise.

For each of the 20 hand tracking markers, create a polynomial fit using 143 tracked
points. Then, use the polynomial fit to resample the points.
"""

def process_all_data(data) -> Dict:
    # num_unique_tracking_points = len(next(iter(data["measurements"].values()))["vr_tracking_points"])
    data_copy = copy.deepcopy(data)

    source_points = np.array([value["setpoint"] for value in data_copy["measurements"].values()])
    tracking_points = [value["vr_tracking_points"] for value in data_copy["measurements"].values()]
    # Loop through 20 times
    for i, unique_tracking_point_array in enumerate(zip(*tracking_points)):
        # This np array has size: 143 x 3
        unique_tracking_point_array = np.array(unique_tracking_point_array)
        poly_mapper = PolynomialMapper(degree=4)
        poly_result = poly_mapper.fit_map(source_points, unique_tracking_point_array, source_points)
        # Now loop through all the measurements and replace the tracking points at index i with the new points
        for j, value in enumerate(data_copy["measurements"].values()):
            value["vr_tracking_points"][i] = poly_result[j].tolist()
    return data_copy

def resample_points(input_file_path: str, output_file_path: str) -> None:
    with open(input_file_path, "r") as f:
        data = json.load(f)
    new_data = process_all_data(data)
    with open(output_file_path, "w") as f:
        json.dump(new_data, f, indent=2)

# Usage example of plotting data
if __name__ == "__main__":
    resample_points("measurement_data.json", "regularized_data.json")

    with open("regularized_data.json", "r") as f:
        data = json.load(f)
    
    # process_all_data(data)
    i = 0
    source_points = []
    for key, value in data["measurements"].items():
        if i < 143:
            source_points.append(value["setpoint"])
        i += 1
    source_points = np.array(source_points)
    print(source_points)

    i = 0
    target_points = []
    finger_points = []
    end_effector_points = []
    for key, value in data["measurements"].items():
        if i < 143:
            target_points.append(value["vr_root_point"])
            end_effector_points.append(value["vr_tracking_points"][3])
        if i % 10 == 0:
            finger_points.extend(value["vr_tracking_points"])
        i += 1
    target_points = np.array(target_points)
    finger_points = np.array(finger_points)
    end_effector_points = np.array(end_effector_points)
    print(target_points)

    poly_mapper = PolynomialMapper(degree=4)
    poly_result = poly_mapper.fit_map(source_points, end_effector_points, source_points)
    print("Polynomial Mapper Result:", poly_result.shape)

    # Visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot source points
    ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='blue', label='Source', alpha=0.6)

    # Plot target points
    ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='red', label='Target', alpha=0.6)

    # Plot finger points
    # ax.scatter(finger_points[:, 0], finger_points[:, 1], finger_points[:, 2], c='green', label='Finger', alpha=0.6)

    # Plot end effector points
    # ax.scatter(poly_result[:, 0], poly_result[:, 1], poly_result[:, 2], c='black', label='End Effector', alpha=0.6)


    # other_root = np.array(data["measurements"]["6_9_1"]["vr_root_point"])
    # ax.scatter(other_root[0], other_root[1], other_root[2], c='yellow', label='Other Root', alpha=0.6)

    # other_points = np.array(data["measurements"]["6_9_1"]["vr_tracking_points"])
    # ax.scatter(other_points[:, 0], other_points[:, 1], other_points[:, 2], c='green', label='Other', alpha=0.6)

    # # Plot new points and their mappings
    # ax.scatter(poly_result[:, 0], poly_result[:, 1], poly_result[:, 2], c='purple', s=100, label='Polynomial Map')

    # # Draw arrows from target points to polynomial mapped points
    # for i in range(len(target_points)):
    #     ax.quiver(target_points[i, 0], target_points[i, 1], target_points[i, 2],
    #               source_points[i, 0] - target_points[i, 0],
    #               source_points[i, 1] - target_points[i, 1],
    #               source_points[i, 2] - target_points[i, 2],
    #               color='green', alpha=0.7, arrow_length_ratio=0.1)

    # # Draw arrows from source to target for a subset of points
    # num_arrows = num_points#min(20, num_points)  # Limit the number of arrows to avoid clutter
    # for i in range(num_arrows):
    #     ax.quiver(source_points[i, 0], source_points[i, 1], source_points[i, 2],
    #               target_points[i, 0] - source_points[i, 0],
    #               target_points[i, 1] - source_points[i, 1],
    #               target_points[i, 2] - source_points[i, 2],
    #               color='gray', alpha=0.5, arrow_length_ratio=0.1)

    # # Draw lines from new points to their mapped positions
    # for i in range(len(new_points)):
    #     ax.plot([new_points[i, 0], poly_result[i, 0]],
    #             [new_points[i, 1], poly_result[i, 1]],
    #             [new_points[i, 2], poly_result[i, 2]], 'purple', linestyle='--')
    #     ax.plot([new_points[i, 0], lerp_result[i, 0]],
    #             [new_points[i, 1], lerp_result[i, 1]],
    #             [new_points[i, 2], lerp_result[i, 2]], 'orange', linestyle='--')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Manifold Point Mapping Visualization')
    ax.legend()

    plt.tight_layout()
    plt.show()