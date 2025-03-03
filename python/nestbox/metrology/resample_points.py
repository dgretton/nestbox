import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nestbox.metrology.manifoldmap import PolynomialMapper, LerpMapper

"""
Resample hand points to smooth out noise.

For each of the 20 hand tracking markers, create a polynomial fit using 143 tracked
points. Then, use the polynomial fit to resample the points.
"""

# Usage example:
if __name__ == "__main__":
    with open("calibration_data.json", "r") as f:
        data = json.load(f)
    
    source_points = []
    for key, value in data["measurements"].items():
        source_points.append(value["vr_tracking_points"][0])
    source_points = np.array(source_points)
    source_points = np.array([data["measurements"]["2_4_1"]["vr_root_point"]])
    print(source_points)

    target_points = []
    for key, value in data["measurements"].items():
        target_points.append(value["vr_tracking_points"][15])
    target_points = np.array(target_points)
    target_points = np.array(data["measurements"]["2_4_1"]["vr_tracking_points"])
    print(target_points)

    # poly_mapper = PolynomialMapper(degree=4)
    # poly_result = poly_mapper.fit_map(source_points, target_points, source_points)
    # print("Polynomial Mapper Result:", poly_result.shape)

    # Visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot source points
    ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='blue', label='Source', alpha=0.6)

    # Plot target points
    ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='red', label='Target', alpha=0.6)

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

    # # Generate some sample 3D data
    # oneaxis = 4
    # num_points = oneaxis * oneaxis * oneaxis
    # # grid from -10 to 10 on all axes
    # source_points = np.stack(np.meshgrid(np.linspace(-10, 10, oneaxis),
    #                             np.linspace(-10, 10, oneaxis),
    #                             np.linspace(-10, 10, oneaxis)), axis=-1).reshape(-1, 3)
    # print(source_points)
    # # Create a non-linear transformation for target points
    # target_points = np.column_stack([
    #     source_points[:, 0] + np.sin(source_points[:, 0]/100) * np.cos(source_points[:, 1]/100) * 20 + 2.5,
    #     source_points[:, 1] + np.sin(source_points[:, 2]/1000) * 100,
    #     source_points[:, 2] + (source_points[:, 0] + source_points[:, 1] + source_points[:, 2]) ** 2 / 300
    # ])

    # # Points to map
    # new_points = np.array([[2.5, 3.5, 4.5], [7.5, 6.5, 5.5], [4.0, 8.0, 2.0], [0, 0, 0]])

    # # Use PolynomialMapper
    # poly_mapper = PolynomialMapper(degree=3)
    # poly_result = poly_mapper.fit_map(source_points, target_points, new_points)
    # print("Polynomial Mapper Result:", poly_result)

    # # Use LerpMapper
    # lerp_mapper = LerpMapper()
    # lerp_result = lerp_mapper.fit_map(source_points, target_points, new_points)
    # print("Lerp Mapper Result:", lerp_result)

    # # Visualization
    # fig = plt.figure(figsize=(12, 10))
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot source points
    # ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='blue', label='Source', alpha=0.6)

    # # Plot target points
    # ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='red', label='Target', alpha=0.6)

    # # Plot new points and their mappings
    # ax.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], c='green', s=100, label='New Points')
    # ax.scatter(poly_result[:, 0], poly_result[:, 1], poly_result[:, 2], c='purple', s=100, label='Polynomial Map')
    # ax.scatter(lerp_result[:, 0], lerp_result[:, 1], lerp_result[:, 2], c='orange', s=100, label='Lerp Map')

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

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Manifold Point Mapping Visualization')
    # ax.legend()

    # plt.tight_layout()
    # plt.show()