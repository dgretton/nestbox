import numpy as np
import pyquaternion
import torch
from typing import Union

def transform_point(origin, quaternion, point):
    origin = coerce_numpy(origin)
    quaternion = coerce_quaternion(quaternion)
    point = coerce_numpy(point)
    return coerce_numpy(quaternion.rotate(point) + np.array(origin))

def inverse_transform_point(origin, quaternion, point):
    origin = coerce_numpy(origin)
    quaternion = coerce_quaternion(quaternion)
    point = coerce_numpy(point)
    return coerce_numpy(quaternion.inverse.rotate(point - origin))

def transform_points(origin, quaternion, points):
    return np.array([transform_point(origin, quaternion, point) for point in points]) # TODO optimize

def inverse_transform_points(origin, quaternion, points):
    return np.array([inverse_transform_point(origin, quaternion, point) for point in points]) # TODO optimize

def rotate_covariance(quaternion, covariance):
    quaternion = coerce_quaternion(quaternion)
    covariance = coerce_numpy(covariance)
    return quaternion.rotation_matrix @ covariance @ quaternion.rotation_matrix.T

def quaternion_to_basis(quaternion):
    quaternion = coerce_quaternion(quaternion)
    return quaternion.rotate([1, 0, 0]), quaternion.rotate([0, 1, 0]), quaternion.rotate([0, 0, 1])

def coerce_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        if tensor.dtype == np.float64:
            return tensor
        return tensor.astype(np.float64)
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().numpy()
    elif isinstance(tensor, pyquaternion.Quaternion):
        return np.array(list(tensor), dtype=np.float64)
    else:
        return np.array(tensor, dtype=np.float64)

def coerce_quaternion(tensor):
    if isinstance(tensor, np.ndarray):
        return pyquaternion.Quaternion(*tensor)
    elif isinstance(tensor, torch.Tensor):
        return pyquaternion.Quaternion(*tensor.detach().numpy())
    elif isinstance(tensor, pyquaternion.Quaternion):
        return tensor
    else:
        return pyquaternion.Quaternion(*tensor)

trinums = {n*(n+1)//2:n for n in range(100)}
maxtrinum = max(trinums)
def upper_triangle_to_covariance(upper_triangle):
    upper_triangle = coerce_numpy(upper_triangle)
    if len(upper_triangle) not in trinums or len(upper_triangle) > maxtrinum:
        raise ValueError(f"Length of upper triangle must be a triangular number less than {maxtrinum}. Got {len(upper_triangle)}")
    n = trinums[len(upper_triangle)]
    cov = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i, n):
            cov[i, j] = upper_triangle[idx]
            cov[j, i] = upper_triangle[idx]
            idx += 1
    return cov

def covariance_to_upper_triangle(covariance):
    covariance = coerce_numpy(covariance)
    n = covariance.shape[0]
    upper_triangle = np.zeros(n*(n+1)//2)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            upper_triangle[idx] = covariance[i, j]
            if not np.isclose(covariance[i, j], covariance[j, i]):
                raise ValueError("Covariance matrix is not symmetric")
            idx += 1
    return upper_triangle


class SE3Transform:
    def __init__(self, position: np.ndarray, orientation: Union[np.ndarray, 'Quaternion']):
        """Create an SE(3) transform from position and orientation.

        Args:
            position: Length 3 array representing translation
            orientation: Either a 3x3 rotation matrix or quaternion
        """
        self.position = np.asarray(position, dtype=np.float64)
        if self.position.shape != (3,):
            raise ValueError(f"Position must be length 3, got shape {self.position.shape}")

        # Convert orientation to 3x3 rotation matrix if needed
        if hasattr(orientation, 'rotation_matrix'):  # Quaternion case
            R = orientation.rotation_matrix
        else:
            R = np.asarray(orientation, dtype=np.float64)
            if R.shape != (3, 3):
                raise ValueError(f"Rotation matrix must be 3x3, got shape {R.shape}")

        # Validate rotation matrix properties
        if not np.allclose(R @ R.T, np.eye(3), rtol=1e-5):
            raise ValueError("Matrix is not orthogonal")
        if not np.allclose(np.linalg.det(R), 1.0, rtol=1e-5):
            raise ValueError("Matrix has determinant != 1")

        # Create and store the 4x4 transform matrix
        self._matrix = np.eye(4)
        self._matrix[:3, :3] = R
        self._matrix[:3, 3] = self.position

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform one or more points by this SE(3) transform.

        Args:
            points: Array of shape (n, 3) or (3,) containing points to transform

        Returns:
            Array of transformed points with same shape as input
        """
        points = np.asarray(points)
        single_point = False
        if points.shape == (3,):
            points = points[np.newaxis, :]
            single_point = True

        # Convert to homogeneous coordinates
        homogeneous = np.ones((len(points), 4))
        homogeneous[:, :3] = points

        # Transform and convert back
        transformed = (self._matrix @ homogeneous.T).T[:, :3]

        return transformed[0] if single_point else transformed

    def compose(self, other: 'SE3Transform') -> 'SE3Transform':
        """Compose this transform with another, returning a new transform."""
        matrix = self._matrix @ other._matrix
        return SE3Transform(matrix[:3, 3], matrix[:3, :3])

    @property 
    def rotation_matrix(self) -> np.ndarray:
        """Get the 3x3 rotation component."""
        return self._matrix[:3, :3]

    @property
    def matrix(self) -> np.ndarray:
        """Get the full 4x4 transform matrix."""
        return self._matrix.copy()  # Return copy to prevent modification

    def inverse(self) -> 'SE3Transform':
        """Return the inverse transform."""
        R = self.rotation_matrix
        inv_matrix = np.eye(4) 
        inv_matrix[:3, :3] = R.T
        inv_matrix[:3, 3] = -R.T @ self.position
        return SE3Transform(inv_matrix[:3, 3], inv_matrix[:3, :3])
