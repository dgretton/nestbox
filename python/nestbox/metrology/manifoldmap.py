import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Type
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.interpolate import LinearNDInterpolator
from scipy.linalg import orthogonal_procrustes
from pyquaternion import Quaternion

from ..numutil import SE3Transform

class ManifoldPointMapper(ABC):
    def __init__(self):
        self.source_points = None
        self.target_points = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, source_points: np.ndarray, target_points: np.ndarray) -> None:
        """
        Fit the mapper to the given point pairs.

        :param source_points: Array of points in the source space
        :param target_points: Corresponding points in the target space
        """
        pass

    @abstractmethod
    def map(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from source space to target space.

        :param points: Array of points to map
        :return: Transformed points
        """
        pass

    def fit_map(self, source_points: np.ndarray, target_points: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Fit the mapper and then map points in one operation.

        :param source_points: Array of points in the source space for fitting
        :param target_points: Corresponding points in the target space for fitting
        :param points: Points to map
        :return: Transformed points
        """
        self.fit(source_points, target_points)
        return self.map(points)


class PolynomialMapper(ManifoldPointMapper):
    def __init__(self, degree: int = 2):
        super().__init__()
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.model = LinearRegression()

    def fit(self, source_points: np.ndarray, target_points: np.ndarray) -> None:
        self.source_points = source_points
        self.target_points = target_points

        # Transform the source points into polynomial features
        X_poly = self.poly_features.fit_transform(source_points)

        # Fit a linear regression model on the polynomial features
        self.model.fit(X_poly, target_points)

        self.is_fitted = True

    def map(self, points: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Mapper must be fitted before map can be called.")

        # Transform the input points into polynomial features
        X_poly = self.poly_features.transform(points)

        # Use the fitted model to predict the target points
        return self.model.predict(X_poly)


class LerpMapper(ManifoldPointMapper):
    def __init__(self):
        super().__init__()
        self.interpolator = None

    def fit(self, source_points: np.ndarray, target_points: np.ndarray) -> None:
        self.source_points = source_points
        self.target_points = target_points

        # Create a linear N-D interpolator
        self.interpolator = LinearNDInterpolator(source_points, target_points)

        self.is_fitted = True

    def map(self, points: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Mapper must be fitted before map can be called.")

        return self.interpolator(points)


class PoseExtractor(ABC):
    @abstractmethod
    def extract_pose(self, point_set: np.ndarray) -> SE3Transform:
        """Extract a pose from a set of points.
        
        Args:
            point_set: Array of shape (n_points, 3) containing point positions
            
        Returns:
            SE3Transform representing the pose that best explains these points
        """
        pass

class ProcrustesBasedPoseExtractor(PoseExtractor):
    def __init__(self, reference_points: np.ndarray):
        """Initialize with the canonical/reference point arrangement.
        
        Args:
            reference_points: Array of shape (n_points, 3) representing the 
                            expected/canonical arrangement of points
        """
        self.reference_points = reference_points
        self._ref_centroid = np.mean(reference_points, axis=0)
        self._centered_reference = reference_points - self._ref_centroid

    def extract_pose(self, point_set: np.ndarray) -> SE3Transform:
        centroid = np.mean(point_set, axis=0)
        centered_points = point_set - centroid
        
        # Find rotation matrix using Procrustes
        R, _ = orthogonal_procrustes(centered_points, self._centered_reference)

        # check if the rotation matrix is special orthogonal
        if not np.allclose(np.linalg.det(R), 1.0, rtol=1e-5):
            raise ValueError("Matrix has determinant != 1")

        return SE3Transform(centroid, R)


class ManifoldPoseMapper(ABC):
    @abstractmethod
    def interpolate(self, query_point: np.ndarray) -> SE3Transform:
        pass

class DataDrivenPoseMapper(ManifoldPoseMapper):
    def __init__(self, 
                 sample_data: Dict[str, np.ndarray],
                 point_interpolator_class: Type[ManifoldPointMapper],
                 pose_extractor: PoseExtractor):
        """
        Args:
            sample_data: Dictionary containing:
                'sample_positions': array of shape (n_samples, 3)
                'point_sets': array of shape (n_samples, n_points, 3)
            point_interpolator_class: Class to use for interpolating points
            pose_extractor: Instance of PoseExtractor to convert point sets to poses
        """
        if not {'sample_positions', 'point_sets'}.issubset(sample_data.keys()):
            raise ValueError("sample_data must contain 'sample_positions' and 'point_sets'")
            
        self.sample_positions = sample_data['sample_positions']
        self.point_sets = sample_data['point_sets']
        self.pose_extractor = pose_extractor
        
        # Create an interpolator for each tracked point
        n_points = self.point_sets.shape[1]
        self.point_interpolators = []
        for i in range(n_points):
            # Extract all positions of point i across all samples
            point_positions = self.point_sets[:, i, :]
            interpolator = point_interpolator_class(self.sample_positions, point_positions)
            self.point_interpolators.append(interpolator)

    def interpolate(self, query_point: np.ndarray) -> SE3Transform:
        # Interpolate each point's position at the query location
        interpolated_points = np.zeros((len(self.point_interpolators), 3))
        for i, interpolator in enumerate(self.point_interpolators):
            interpolated_points[i] = interpolator.interpolate(query_point)
            
        # Extract pose from interpolated point set
        return self.pose_extractor.extract_pose(interpolated_points)


class DirectPoseMapper(ManifoldPoseMapper):
    def __init__(self, sample_positions: np.ndarray, poses: List[SE3Transform]):
        self.sample_positions = np.asarray(sample_positions)
        self.poses = poses
        if len(sample_positions) != len(poses):
            raise ValueError("Must have same number of positions and poses")
        
        # For interpolating positions
        self.position_interpolator = LinearNDInterpolator(
            self.sample_positions,
            np.array([pose.position for pose in poses])
        )

        # For finding nearest neighbors
        from scipy.spatial import cKDTree
        self.kdtree = cKDTree(self.sample_positions)

    def _rotation_to_lie_algebra(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to lie algebra representation (rotation vector).
        
        Args:
            R: 3x3 rotation matrix
        Returns:
            3D rotation vector (axis * angle)
        """
        # Rotation angle from trace
        theta = np.arccos((np.trace(R) - 1) / 2)
        
        if np.abs(theta) < 1e-10:  # Identity rotation
            return np.zeros(3)
        
        # Rotation axis from skew-symmetric part
        axis = np.array([
            R[2,1] - R[1,2],
            R[0,2] - R[2,0],
            R[1,0] - R[0,1]
        ]) / (2 * np.sin(theta))
        
        return theta * axis

    def _lie_algebra_to_rotation(self, w: np.ndarray) -> np.ndarray:
        """Convert lie algebra element (rotation vector) to rotation matrix.
        
        Args:
            w: 3D rotation vector
        Returns:
            3x3 rotation matrix
        """
        theta = np.linalg.norm(w)
        if theta < 1e-10:
            return np.eye(3)
            
        axis = w / theta
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        
        return (np.eye(3) + np.sin(theta) * K + 
                (1 - np.cos(theta)) * (K @ K))

    def _check_rotation_distances(self, rotations: List[np.ndarray]) -> float:
        """Check maximum angular difference between any pair of rotations.
        
        Returns:
            Maximum angle between any two rotations in radians
        """
        max_angle = 0
        for i, R1 in enumerate(rotations):
            for R2 in rotations[i+1:]:
                # Relative rotation R1^T @ R2
                R_diff = R1.T @ R2
                angle = np.arccos((np.trace(R_diff) - 1) / 2)
                max_angle = max(max_angle, angle)
        return max_angle

    def interpolate(self, query_point: np.ndarray) -> SE3Transform:
        # Interpolate position
        position = self.position_interpolator(query_point)
        
        # Find k nearest neighbors
        k = min(4, len(self.poses))
        distances, indices = self.kdtree.query(query_point, k=k)
        
        # Get rotations for nearest poses
        rotations = [self.poses[i].rotation_matrix for i in indices]
        
        # Check if rotations are similar enough
        max_angle = self._check_rotation_distances(rotations)
        if max_angle > np.pi/2:
            raise ValueError(
                f"Rotations differ by {max_angle:.2f} radians, "
                f"which exceeds safe threshold of π/2. "
                f"Interpolation may be unreliable."
            )
        
        # Convert to lie algebra
        rot_vecs = [self._rotation_to_lie_algebra(R) for R in rotations]
        
        # Calculate weights
        eps = 1e-10
        distances = np.maximum(distances, eps)
        weights = 1.0 / distances
        weights = weights / np.sum(weights)
        
        # Weighted average in lie algebra
        avg_vec = np.sum([w * vec for w, vec in zip(weights, rot_vecs)], axis=0)
        
        # Convert back to rotation matrix
        rotation = self._lie_algebra_to_rotation(avg_vec)
        
        return SE3Transform(position, rotation)


# Usage example:
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Generate some sample 3D data
    oneaxis = 4
    num_points = oneaxis * oneaxis * oneaxis
    # grid from -10 to 10 on all axes
    source_points = np.stack(np.meshgrid(np.linspace(-10, 10, oneaxis),
                                np.linspace(-10, 10, oneaxis),
                                np.linspace(-10, 10, oneaxis)), axis=-1).reshape(-1, 3)
    print(source_points)
    # Create a non-linear transformation for target points
    target_points = np.column_stack([
        source_points[:, 0] + np.sin(source_points[:, 0]/100) * np.cos(source_points[:, 1]/100) * 20 + 2.5,
        source_points[:, 1] + np.sin(source_points[:, 2]/1000) * 100,
        source_points[:, 2] + (source_points[:, 0] + source_points[:, 1] + source_points[:, 2]) ** 2 / 300
    ])

    # Points to map
    new_points = np.array([[2.5, 3.5, 4.5], [7.5, 6.5, 5.5], [4.0, 8.0, 2.0], [0, 0, 0]])

    # Use PolynomialMapper
    poly_mapper = PolynomialMapper(degree=3)
    poly_result = poly_mapper.fit_map(source_points, target_points, new_points)
    print("Polynomial Mapper Result:", poly_result)

    # Use LerpMapper
    lerp_mapper = LerpMapper()
    lerp_result = lerp_mapper.fit_map(source_points, target_points, new_points)
    print("Lerp Mapper Result:", lerp_result)

    # Visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot source points
    ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='blue', label='Source', alpha=0.6)

    # Plot target points
    ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='red', label='Target', alpha=0.6)

    # Plot new points and their mappings
    ax.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], c='green', s=100, label='New Points')
    ax.scatter(poly_result[:, 0], poly_result[:, 1], poly_result[:, 2], c='purple', s=100, label='Polynomial Map')
    ax.scatter(lerp_result[:, 0], lerp_result[:, 1], lerp_result[:, 2], c='orange', s=100, label='Lerp Map')

    # Draw arrows from source to target for a subset of points
    num_arrows = num_points#min(20, num_points)  # Limit the number of arrows to avoid clutter
    for i in range(num_arrows):
        ax.quiver(source_points[i, 0], source_points[i, 1], source_points[i, 2],
                  target_points[i, 0] - source_points[i, 0],
                  target_points[i, 1] - source_points[i, 1],
                  target_points[i, 2] - source_points[i, 2],
                  color='gray', alpha=0.5, arrow_length_ratio=0.1)

    # Draw lines from new points to their mapped positions
    for i in range(len(new_points)):
        ax.plot([new_points[i, 0], poly_result[i, 0]],
                [new_points[i, 1], poly_result[i, 1]],
                [new_points[i, 2], poly_result[i, 2]], 'purple', linestyle='--')
        ax.plot([new_points[i, 0], lerp_result[i, 0]],
                [new_points[i, 1], lerp_result[i, 1]],
                [new_points[i, 2], lerp_result[i, 2]], 'orange', linestyle='--')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Manifold Point Mapping Visualization')
    ax.legend()

    plt.tight_layout()
    plt.show()