"""Deterministic online aligner: damped Gauss-Newton on SE(3), with features as latent points.

Model. Coordinate system i has pose (R_i, t_i): p_world = R_i p_local + t_i (same convention as the other aligners:
current_origins[i] = t_i, current_orientations[i] = w-first quaternion of R_i). Every feature seen by two or more
coordinate systems gets a latent world position x_f. A NormalMeasurement (mu, Sigma) of feature f in CS i contributes
the whitened residual

    r = W (R_i^T (x_f - t_i) - mu),    W^T W = Sigma^-1,

so anisotropic covariances (long ovals, e.g. camera rays) are honored exactly. The objective, 1/2 sum |r|^2, is the
joint MAP over poses and feature positions. (The marginal likelihood with x_f integrated out differs by a log-det
term that rewards near-parallel, i.e. less informative, ray geometries; it is deliberately left out.)

Dynamics. Each step: place newly appearing (or badly misplaced) coordinate systems in closed form (weighted Kabsch
against neighbors, adopted only when far better than the current pose), set every x_f to its closed-form optimum given the poses, then move a fraction alpha of a damped
Gauss-Newton step. alpha = 1 - exp(-dt / convergence_time) depends only on the wall-clock time since the previous
step, so the dynamics are autonomous: no schedule, no step counter, no momentum. Restarting from a saved estimate is
seamless, data changes are picked up on the next step, and with static data the estimate converges to a fixed point
and stays there. Directions the data doesn't constrain get no gradient, so they stay put instead of random-walking.

Solver. Feature points are eliminated (Schur complement), so each step solves a 6(N-1) pose system: cost grows with
the number of coordinate systems, not features.

Uncertainty. pose_covariances[name] is the Laplace approximation (inverse Gauss-Newton Hessian, marginalized over
everything else) for each coordinate system relative to its component's fixed CS: 6x6, ordered
[body-frame rotation (rad, applied as R <- R Exp(phi)), world-frame translation (m)]. feature_estimates[f] is each
shared feature's world position and posterior covariance; measurement_ellipsoids() gives every individual measurement
as a world-frame Gaussian under the current poses.
"""
import time
import numpy as np
from .aligner import Aligner
from ..measurement import NormalMeasurement
from ..numutil import coerce_numpy


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _exp_so3(phi):
    theta = np.linalg.norm(phi)
    K = _skew(phi)
    if theta < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + np.sin(theta) / theta * K + (1 - np.cos(theta)) / theta ** 2 * (K @ K)


def _log_so3(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    theta = np.arccos(c)
    if theta < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    if np.pi - theta < 1e-6:  # near 180 degrees: axis from the symmetric part
        axis = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        k = int(np.argmax(axis))
        axis = (R[:, k] + np.eye(3)[:, k]) / np.linalg.norm(R[:, k] + np.eye(3)[:, k])
        return theta * axis
    return theta / (2 * np.sin(theta)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def _quat_to_matrix(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_quat(R):
    # Shepperd's method, w-first, w >= 0
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def _whitener(cov, min_variance=1e-12):
    """W with W^T W = cov^-1 (robust to near-singular and very elongated covariances)."""
    evals, evecs = np.linalg.eigh((cov + cov.T) / 2)
    evals = np.maximum(evals, min_variance)
    return (evecs / np.sqrt(evals)).T


class GaussNewtonAligner(Aligner):
    def __init__(self, convergence_time=1.0, step_interval=0.02, damping=1e-9, **kwargs):
        super().__init__()
        self.convergence_time = convergence_time  # seconds for ~63% of the way to the optimum; <= 0 means full steps
        self.step_interval = step_interval        # run_optimizer sleeps this long between steps
        self.damping = damping                    # Levenberg-Marquardt damping, relative to the Hessian diagonal
        self.sampled_points = np.zeros((0, 3))    # world feature estimates (the visualizer reads this attribute)
        self.pose_covariances = {}
        self.feature_estimates = {}  # feature -> (world position, posterior covariance) for features seen by >= 2 CSs
        self.cost = None
        self._last_step = None

    # ---------- inspection ----------

    def measurement_ellipsoids(self, cs_names=None):
        """Every NormalMeasurement as a world-frame Gaussian under the current pose estimates:
        [(cs name, feature, center, covariance)], center = R mu + t, covariance = R Sigma R^T. Includes measurements of
        features no other CS sees. Render e.g. a k-sigma ellipsoid from eigh(covariance)."""
        out = []
        with self.lock:
            for i, cs in enumerate(self.coordinate_systems):
                if cs_names is not None and cs.name not in cs_names:
                    continue
                R, t = _quat_to_matrix(self.current_orientations[i]), coerce_numpy(self.current_origins[i]).astype(float)
                for feature, meas in list(cs.measurements.items()):
                    if not isinstance(meas, NormalMeasurement):
                        continue
                    mu = coerce_numpy(meas.mean).astype(float).reshape(-1)
                    cov = coerce_numpy(meas.covariance).astype(float)
                    if mu.shape == (3,) and cov.shape == (3, 3):
                        out.append((cs.name, feature, R @ mu + t, R @ cov @ R.T))
        return out

    # ---------- problem assembly ----------

    def _poses(self):
        Rs = [_quat_to_matrix(q) for q in self.current_orientations]
        ts = [coerce_numpy(o).astype(float) for o in self.current_origins]
        return Rs, ts

    def _observations(self):
        """{feature: [(cs_idx, mu, W, P)]} for NormalMeasurements of features seen by >= 2 coordinate systems."""
        by_feature = {}
        for i, cs in enumerate(self.coordinate_systems):
            for feature, meas in list(cs.measurements.items()):
                if not isinstance(meas, NormalMeasurement):
                    continue
                mu = coerce_numpy(meas.mean).astype(float).reshape(-1)
                cov = coerce_numpy(meas.covariance).astype(float)
                if mu.shape != (3,) or cov.shape != (3, 3):
                    continue
                W = _whitener(cov)
                by_feature.setdefault(feature, []).append((i, mu, W, W.T @ W))
        return {f: obs for f, obs in by_feature.items() if len({o[0] for o in obs}) >= 2}

    def _components(self, obs):
        parent = list(range(len(self.coordinate_systems)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for o in obs.values():
            for k in o[1:]:
                parent[find(k[0])] = find(o[0][0])
        comps = {}
        involved = {k[0] for o in obs.values() for k in o}
        for i in involved:
            comps.setdefault(find(i), []).append(i)
        return [sorted(c) for c in comps.values()]

    @staticmethod
    def _landmark(obs_f, Rs, ts, members=None):
        """Closed-form optimal world position of a feature given poses (precision-weighted)."""
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for i, mu, _, P in obs_f:
            if members is not None and i not in members:
                continue
            Pw = Rs[i] @ P @ Rs[i].T
            A += Pw
            b += Pw @ (Rs[i] @ mu + ts[i])
        if not A.any():
            return None
        return np.linalg.pinv(A) @ b

    # ---------- closed-form placement of new coordinate systems ----------

    def _place(self, obs, comps, Rs, ts, fixed, alpha):
        """Closed-form placement, memoryless: walk outward from each component's fixed CS; for every other CS,
        compute a weighted-Kabsch pose against the already-visited neighbors and adopt it only if it is far
        better (< 1/4 the local cost) than the CS's current pose, moving a fraction alpha of the way there along the
        geodesic (same time constant as the Gauss-Newton steps, so a new CS drifts in rather than snapping). A CS
        that is already near the optimum (e.g. after a restart from a saved estimate) is left alone."""
        for comp in comps:
            visited = {i for i in comp if i in fixed}
            progress = True
            while progress:
                progress = False
                for u in comp:
                    if u in visited:
                        continue
                    pairs = []  # (mu_u, W_u, P_u, landmark from visited observers)
                    for obs_f in obs.values():
                        mine = [k for k in obs_f if k[0] == u]
                        if not mine or not any(k[0] in visited for k in obs_f):
                            continue
                        _, mu, W, P = mine[0]
                        pairs.append((mu, W, P, self._landmark(obs_f, Rs, ts, members=visited)))
                    if not pairs:
                        continue
                    visited.add(u)
                    progress = True

                    def local_cost(R, t):
                        return sum(float(np.sum((W @ (R.T @ (x - t) - mu)) ** 2)) for mu, W, _, x in pairs)
                    local = np.array([p[0] for p in pairs])
                    world = np.array([p[3] for p in pairs])
                    w = np.array([1.0 / np.trace(np.linalg.pinv(p[2])) for p in pairs])
                    w = w / w.sum()
                    cl, cw = w @ local, w @ world
                    R_c = Rs[u]
                    if len(pairs) >= 3 and np.linalg.svd(local - cl, compute_uv=False)[1] > 1e-6:
                        U, _, Vt = np.linalg.svd(((local - cl) * w[:, None]).T @ (world - cw))
                        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
                        R_c = Vt.T @ D @ U.T
                    t_c = cw - R_c @ cl  # with < 3 usable points only the translation is proposed
                    if local_cost(R_c, t_c) < 0.25 * local_cost(Rs[u], ts[u]):
                        Rs[u] = Rs[u] @ _exp_so3(alpha * _log_so3(Rs[u].T @ R_c))
                        ts[u] = ts[u] + alpha * (t_c - ts[u])

    # ---------- one step ----------

    def _cost(self, obs, Rs, ts):
        c = 0.0
        for obs_f in obs.values():
            x = self._landmark(obs_f, Rs, ts)
            for i, mu, W, _ in obs_f:
                r = W @ (Rs[i].T @ (x - ts[i]) - mu)
                c += 0.5 * r @ r
        return c

    def _build_model(self):
        pass  # nothing to build; the problem is reassembled from the current measurements every step

    def _gradient_descent_step(self):
        now = time.monotonic()
        dt = self.step_interval if self._last_step is None else now - self._last_step
        self._last_step = now
        alpha = 1.0 if self.convergence_time <= 0 else 1.0 - np.exp(-dt / self.convergence_time)

        try:
            obs = self._observations()
        except RuntimeError:  # measurements changed mid-iteration on another thread; try next step
            return
        if not obs:
            return
        Rs, ts = self._poses()
        comps = self._components(obs)
        # fixed CS per component (it stays where it is): the pinned one if present, else the lowest index
        fixed = {self.pinned_cs_idx if self.pinned_cs_idx in comp else comp[0] for comp in comps}
        self._place(obs, comps, Rs, ts, fixed, alpha)
        free = [i for comp in comps for i in comp if i not in fixed]
        features = list(obs)
        pose_col = {i: 6 * k for k, i in enumerate(free)}
        n = 6 * len(free)

        # Gauss-Newton system with every feature point eliminated (Schur complement). Each residual touches one pose and
        # one feature, so the pose-pose block is block-diagonal and each feature is an independent 3x3 block:
        #   S = H_pp - sum_f B_f H_ff^-1 B_f^T,   b = g_p - sum_f B_f H_ff^-1 g_f
        # Cost scales with the number of coordinate systems, not features.
        S = np.zeros((n, n))
        b = np.zeros(n)
        landmarks, eliminated = {}, {}
        for f in features:
            x = self._landmark(obs[f], Rs, ts)
            landmarks[f] = x
            Hff = np.zeros((3, 3))
            gf = np.zeros(3)
            Bf = []  # (pose column, 6x3 cross block)
            for i, mu, W, _ in obs[f]:
                v = Rs[i].T @ (x - ts[i])
                r = W @ (v - mu)
                Jl = W @ Rs[i].T
                Hff += Jl.T @ Jl
                gf += Jl.T @ r
                if i in pose_col:
                    c = pose_col[i]
                    Jp = np.hstack([W @ _skew(v), -Jl])
                    S[c:c + 6, c:c + 6] += Jp.T @ Jp
                    b[c:c + 6] += Jp.T @ r
                    Bf.append((c, Jp.T @ Jl))
            Kf = np.linalg.inv(Hff + 1e-12 * max(np.trace(Hff), 1.0) * np.eye(3))
            for ca, Ba in Bf:
                b[ca:ca + 6] -= Ba @ Kf @ gf
                for cb, Bb in Bf:
                    S[ca:ca + 6, cb:cb + 6] -= Ba @ Kf @ Bb.T
            eliminated[f] = (Kf, Bf)
        self.sampled_points = np.array([landmarks[f] for f in features])

        if n:
            diag = np.diag(S).copy()
            lam = self.damping * diag + 1e-12 * max(diag.max(), 1.0)
            delta = -np.linalg.solve(S + np.diag(lam), b)
        else:
            delta = np.zeros(0)

        cost0 = self._cost(obs, Rs, ts)
        for _ in range(8):  # memoryless backtracking: shrink this step only if it would increase the cost
            Rn, tn = list(Rs), list(ts)
            for i, c in pose_col.items():
                Rn[i] = Rs[i] @ _exp_so3(alpha * delta[c:c + 3])
                tn[i] = ts[i] + alpha * delta[c + 3:c + 6]
            cost1 = self._cost(obs, Rn, tn)
            if cost1 <= cost0 + 1e-12 * max(cost0, 1.0):
                break
            alpha *= 0.5
        else:
            Rn, tn, cost1 = Rs, ts, cost0
        self.cost = cost1

        # Laplace covariances. The inverse of the reduced system is exactly the pose block of the full inverse (poses
        # marginalized over features). Inverting the (barely) damped matrix, not a pseudo-inverse, so unconstrained
        # directions show up as huge variance rather than as zero. Feature-point posteriors follow per feature:
        #   Cov(x_f) = K_f + K_f B_f^T Cov(poses) B_f K_f
        Spp = np.linalg.inv(S + np.diag(lam)) if n else np.zeros((0, 0))
        self.pose_covariances = {self.coordinate_systems[i].name: Spp[c:c + 6, c:c + 6] for i, c in pose_col.items()}
        for i in fixed:
            self.pose_covariances[self.coordinate_systems[i].name] = np.zeros((6, 6))
        feature_covariances = {}
        for f, (Kf, Bf) in eliminated.items():
            C = Kf.copy()
            for ca, Ba in Bf:
                for cb, Bb in Bf:
                    C += Kf @ Ba.T @ Spp[ca:ca + 6, cb:cb + 6] @ Bb @ Kf
            feature_covariances[f] = C

        # pinned gauge: express everything so the pinned CS sits at the identity (as the other aligners do)
        if self.pinned_cs_idx is not None and self.pinned_cs_idx < len(Rn):
            Rp, tp = Rn[self.pinned_cs_idx], tn[self.pinned_cs_idx]
            Rn = [Rp.T @ R for R in Rn]
            tn = [Rp.T @ (t - tp) for t in tn]
            self.sampled_points = (self.sampled_points - tp) @ Rp if len(self.sampled_points) else self.sampled_points
            feature_covariances = {f: Rp.T @ C @ Rp for f, C in feature_covariances.items()}
        self.current_orientations = [_matrix_to_quat(R) for R in Rn]
        self.current_origins = list(tn)
        self.feature_estimates = {f: (p, feature_covariances[f]) for f, p in zip(features, self.sampled_points)}
        for cs in self.coordinate_systems:
            cs.set_stale(False)
