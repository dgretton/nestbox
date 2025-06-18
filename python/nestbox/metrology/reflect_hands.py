import json

import numpy as np
import matplotlib.pyplot as plt

def reflect_measurement_data(measurement_data):
    for number, inner_data in measurement_data["measurements"].items():
        inner_data["vr_root_point"][2] *= -1    # Reflect root point across the x-y plane
        for vr_tracking_point in inner_data["vr_tracking_points"]:
            vr_tracking_point[2] *= -1    # Reflect tracking points across the x-y plane
    return measurement_data

if __name__ == "__main__":
    with open("angled_hand_mount_hand_positions.json", "r") as f:
        data = json.load(f)
    print(data["hand_points"])
    hand_points = np.array(data["hand_points"])
    # hand_points[:, 2] *= -1   TODO: reflect points taken from VR measurement

    # Reflect VR measurement data across the x-y plane to convert from left-hand to right-hand coordinate system
    with open("measurement_data.json", "r") as f:
        measurement_data = json.load(f)
    print(measurement_data)
    measurement_data = reflect_measurement_data(measurement_data)
    with open("measurement_data_reflected.json", "w") as f:
        json.dump(measurement_data, f, indent=4)

    # Visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot source points
    ax.scatter(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2], c='blue', label='Source', alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Manifold Point Mapping Visualization')
    ax.legend()

    plt.tight_layout()
    plt.show()
