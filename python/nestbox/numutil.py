import numpy as np
import pyquaternion
import torch

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
        if tensor.dtype == np.float32:
            return tensor
        return tensor.astype(np.float32)
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().numpy()
    elif isinstance(tensor, pyquaternion.Quaternion):
        return np.array(list(tensor), dtype=np.float32)
    else:
        return np.array(tensor, dtype=np.float32)

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
