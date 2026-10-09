"""Local-coordinate residual blocks for the deterministic aligner."""
import numpy as np


def _whitener(cov, min_variance=1e-12):
    """W with W.T W = cov^-1, robust to elongated Gaussian covariances."""
    evals, evecs = np.linalg.eigh((cov + cov.T) / 2)
    return (evecs / np.sqrt(np.maximum(evals, min_variance))).T


class GaussianResidual:
    dim = 3
    metric = True

    def __init__(self, mean, covariance):
        self.mu = mean
        self.W = _whitener(covariance)
        self.P = self.W.T @ self.W

    def evaluate(self, v):
        return self.W @ (v - self.mu), self.W

    def triangulation_terms(self, R, t):
        A = R @ self.P @ R.T
        return A, A @ (R @ self.mu + t)


class BearingResidual:
    def __init__(self, measurement):
        self.meas = measurement
        self.L = _whitener(measurement.angular_covariance)
        self.dim = 2 + (measurement.depth_prior is not None)
        self.metric = measurement.depth_prior is not None

    def evaluate(self, v):
        m = self.meas
        w = v - m.origin
        rho = np.linalg.norm(w)
        z = m.direction @ w
        if z <= 1e-3 * rho or rho == 0:
            return None  # behind the sensor or too close to the image-plane singularity
        y = m.basis.T @ w
        r = self.L @ y / z
        J = self.L @ (m.basis.T / z - np.outer(y, m.direction) / z ** 2)
        if m.depth_prior is not None:
            rho0, sigma = m.depth_prior
            r = np.append(r, np.log(rho / rho0) / sigma)
            J = np.vstack((J, w / (rho ** 2 * sigma)))
        return r, J

    def triangulation_terms(self, R, t):
        m = self.meas
        rho = m.depth_prior[0] if m.depth_prior is not None else 1.0
        # Anisotropic generalization of (I-dd.T)/(sigma_theta*rho)^2.
        W = self.L @ m.basis.T @ R.T / rho
        A = W.T @ W
        center = R @ m.origin + t
        b = A @ center
        if m.depth_prior is not None:
            d = R @ m.direction
            P = np.outer(d, d) / (rho * m.depth_prior[1]) ** 2
            A += P
            b += P @ (center + rho * d)
        return A, b
