import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from nestbox.numutil import SE3Transform
from nestbox.metrology.manifoldmap import DirectPoseMapper

def plot_coordinate_frame(ax, pose, scale=1.0, label=None):
    """Plot a coordinate frame at the given pose."""
    origin = pose.position
    # Get axes directions from rotation matrix
    x_axis = pose.rotation_matrix[:, 0] * scale
    y_axis = pose.rotation_matrix[:, 1] * scale
    z_axis = pose.rotation_matrix[:, 2] * scale
    
    # Plot axes
    ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color='r')
    ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color='g')
    ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color='b')
    
    if label:
        ax.text(origin[0], origin[1], origin[2], label)

def create_simple_test_data(size=1.0):
    """Create 4 poses at the corners of a square in the XY plane."""
    # Create 4 positions at corners of a square
    positions = np.array([
        [-size, -size, -0.2],  # bottom-left
        [size, -size, 0.2],   # bottom-right
        [-size, size, 0.2],   # top-left
        [size, size, -0.2]     # top-right
    ])
    
    # Create poses with varying orientations at each corner
    poses = []
    for i, pos in enumerate(positions):
        # Create a rotation that's different at each corner
        # This example uses a rotation around the z-axis based on position
        angle_z = i * np.pi / 10
        
        # Create rotation matrix
        rot = Rotation.from_euler('xyz', [0, 0, angle_z])
        rot_matrix = rot.as_matrix()
        
        poses.append(SE3Transform(pos, rot_matrix))
    
    return positions, poses

def print_pose_info(pose, label="Pose"):
    """Print pose information in an easy-to-read format."""
    print(f"\n{label}:")
    print(f"  Position: {pose.position}")
    
    # Get Euler angles from rotation matrix (in degrees)
    euler_angles = Rotation.from_matrix(pose.rotation_matrix).as_euler('xyz', degrees=True)
    
    print(f"  Rotation Matrix:\n{pose.rotation_matrix}")
    print(f"  Euler Angles (xyz, degrees): {euler_angles}")
    print("-" * 50)

def main():
    # Create simple test data with 4 poses
    positions, poses = create_simple_test_data(size=1.0)
    
    # Print information for each original pose
    for i, pose in enumerate(poses):
        print_pose_info(pose, label=f"Original Pose {i+1}")
    
    # Create the DirectPoseMapper
    mapper = DirectPoseMapper(positions, poses)
    
    # Define the center point where we want to find the average pose
    center_point = np.array([0, 0, 0])
    
    # Interpolate to get the average pose at the center
    center_pose = mapper.interpolate(center_point)
    
    # Print information for the interpolated pose
    print_pose_info(center_pose, label="Interpolated Average Pose")
    
    # Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original sample points
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', s=100, label='Corner Poses')
    
    # Plot center point
    ax.scatter([center_point[0]], [center_point[1]], [center_point[2]], c='red', s=150, label='Center Point')
    
    # Plot coordinate frames for original poses
    for i, pose in enumerate(poses):
        plot_coordinate_frame(ax, pose, scale=0.3, label=f"P{i}")
    
    # Plot the interpolated center pose
    plot_coordinate_frame(ax, center_pose, scale=0.4, label="Avg")
    
    # Set plot properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Average Pose Interpolation')
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()