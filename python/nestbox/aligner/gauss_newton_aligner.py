"""Deterministic online aligner: damped Gauss-Newton on SE(3), with features as latent points.

Model. Coordinate system i has pose (R_i, t_i): p_world = R_i p_local + t_i (same convention as the other aligners:
current_origins[i] = t_i, current_orientations[i] = w-first quaternion of R_i). Every feature seen by two or more
coordinate systems gets a latent world position x_f. A NormalMeasurement (mu, Sigma) of feature f in CS i contributes
the whitened residual

    r = W (R_i^T (x_f - t_i) - mu),    W^T W = Sigma^-1,

so anisotropic Gaussian covariances are honored exactly. BearingMeasurement instead contributes the
whitened tangent-plane projection E^T(v-origin)/(direction^T(v-origin)), with optional log-range prior.
Behind-camera bearings are skipped and counted in skipped_bearings; backtracking cannot discard active residuals.
The objective, 1/2 sum |r|^2, is the
joint MAP over poses and feature positions. (The marginal likelihood with x_f integrated out differs by a log-det
term that rewards near-parallel, i.e. less informative, ray geometries; it is deliberately left out.)

Dynamics. Each step: place newly appearing (or badly misplaced) coordinate systems in closed form (weighted Kabsch
against neighbors, with iterative ray depths for bearings, adopted only when far better than the current pose),
solve every x_f conditionally given the poses (closed form for Gaussians, triangulation plus bounded GN for bearings),
then move a fraction alpha of a damped
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
as a world-frame Gaussian under the current poses, while measurement_cones() exposes bearing uncertainty.
Components without Gaussian measurements or depth priors warn once about the unobservable scale.
"""
import time
import warnings
import numpy as np
from .aligner import Aligner
from ..measurement import NormalMeasurement, BearingMeasurement
from .measurement_residual import GaussianResidual, BearingResidual, _whitener
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
        self.skipped_bearings = 0
        self._warned_scale_components = set()

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

    def measurement_cones(self, k=1, cs_names=None, include_basis=False):
        """World cones: (cs, feature, apex, axis, half_angles, length).

        Angles are k-sigma along the principal covariance axes. With include_basis=True,
        append their world 3x2 basis, needed to render anisotropic/correlated cones.
        Length is 1.5 times shared feature range, otherwise prior range, otherwise 2 m.
        """
        if not np.isfinite(k) or k < 0:
            raise ValueError('k must be finite and nonnegative')
        out = []
        with self.lock:
            Rs, ts = self._poses()
            for i, cs in enumerate(self.coordinate_systems):
                if cs_names is not None and cs.name not in cs_names:
                    continue
                for feature, m in list(cs.measurements.items()):
                    if not isinstance(m, BearingMeasurement):
                        continue
                    apex, axis = Rs[i] @ m.origin + ts[i], Rs[i] @ m.direction
                    values, vectors = np.linalg.eigh(m.angular_covariance)
                    if feature in self.feature_estimates:
                        length = 1.5 * np.linalg.norm(self.feature_estimates[feature][0] - apex)
                    else:
                        length = m.depth_prior[0] if m.depth_prior is not None else 2.0
                    row = (cs.name, feature, apex, axis, k * np.sqrt(values), length)
                    out.append(row + (Rs[i] @ m.basis @ vectors,) if include_basis else row)
        return out

    # ---------- problem assembly ----------

    def _poses(self):
        Rs = [_quat_to_matrix(q) for q in self.current_orientations]
        ts = [coerce_numpy(o).astype(float) for o in self.current_origins]
        return Rs, ts

    def _observations(self):
        """{feature: [(cs_idx, residual)]} for shared 3-D points and bearings."""
        by_feature = {}
        for i, cs in enumerate(self.coordinate_systems):
            for feature, meas in list(cs.measurements.items()):
                if isinstance(meas, BearingMeasurement):
                    residual = BearingResidual(meas)
                elif isinstance(meas, NormalMeasurement):
                    mu = coerce_numpy(meas.mean).astype(float).reshape(-1)
                    cov = coerce_numpy(meas.covariance).astype(float)
                    if mu.shape != (3,) or cov.shape != (3, 3):
                        continue
                    residual = GaussianResidual(mu, cov)
                else:
                    continue
                by_feature.setdefault(feature, []).append((i, residual))
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
        """Deterministic triangulation followed by a bounded feature-only GN solve.

        No cached dynamics: the same poses and observations give the same result on restart.
        Gaussian-only features retain the original precision-weighted closed form exactly.
        """
        selected = [(i, m) for i, m in obs_f if members is None or i in members]
        A, b = np.zeros((3, 3)), np.zeros(3)
        for i, m in selected:
            Ai, bi = m.triangulation_terms(Rs[i], ts[i])
            A += Ai
            b += bi
        if not A.any():
            return None
        x = np.linalg.pinv(A) @ b
        if all(isinstance(m, GaussianResidual) for _, m in selected):
            return x
        if np.linalg.eigvalsh(A)[0] <= 1e-10 * np.trace(A):
            if members is not None:
                return None  # a single un-ranged ray cannot place another coordinate system
            # Choose a finite depth along the unobservable direction, without imposing a prior.
            seeds = [Rs[i] @ (m.meas.origin + m.meas.direction *
                     (m.meas.depth_prior[0] if m.metric else 1.0)) + ts[i]
                     for i, m in selected if isinstance(m, BearingResidual)]
            x += (np.eye(3) - np.linalg.pinv(A) @ A) @ np.mean(seeds, axis=0)

        def evaluate(point):
            return [(i, m.evaluate(Rs[i].T @ (point - ts[i]))) for i, m in selected]

        for _ in range(12):
            rows = evaluate(x)
            valid = [(i, result) for i, result in rows if result is not None]
            if not valid:
                break
            H, g = np.zeros((3, 3)), np.zeros(3)
            cost = 0.0
            for i, (r, D) in valid:
                J = D @ Rs[i].T
                H += J.T @ J
                g += J.T @ r
                cost += r @ r
            dx = -np.linalg.solve(H + 1e-12 * max(np.trace(H), 1.0) * np.eye(3), g)
            # Near-parallel rays must not send the feature off to infinity in one step.
            bound = max(1.0, min(np.linalg.norm(x - ts[i]) for i, _ in valid)) * 0.5
            dx *= min(1.0, bound / max(np.linalg.norm(dx), 1e-30))
            if np.linalg.norm(dx) < 1e-12 * max(1.0, np.linalg.norm(x)):
                break
            for _ in range(10):
                trial = evaluate(x + dx)
                # A step cannot lower cost by moving an active point behind its sensor.
                if all(new is not None for (_, old), (_, new) in zip(rows, trial) if old is not None):
                    trial_cost = sum(result[0] @ result[0] for _, result in trial if result is not None)
                    if trial_cost <= cost:
                        x += dx
                        break
                dx *= 0.5
            else:
                break
        return x

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
                    pairs = []  # (residual, landmark from visited metric observers)
                    for obs_f in obs.values():
                        mine = [k for k in obs_f if k[0] == u]
                        if not mine or not any(k[0] in visited for k in obs_f):
                            continue
                        x = self._landmark(obs_f, Rs, ts, members=visited)
                        if x is not None:
                            pairs.append((mine[0][1], x))
                    if not pairs:
                        continue
                    visited.add(u)
                    progress = True

                    # Once enough points are in front, behind-camera outliers must not
                    # provoke re-placement of an otherwise valid pose. When a camera is
                    # first introduced facing away, Kabsch can still initialize it.
                    visible = [(m, x) for m, x in pairs if m.evaluate(Rs[u].T @ (x - ts[u])) is not None]
                    if len(visible) >= 3:
                        pairs = visible

                    def local_cost(R, t):
                        rows = [m.evaluate(R.T @ (x - t)) for m, x in pairs]
                        if any(row is None for row in rows):
                            return np.inf
                        return sum(float(r @ r) for r, _ in rows)

                    world = np.array([x for _, x in pairs])
                    median_depth = max(float(np.median(np.linalg.norm(world - ts[u], axis=1))), 0.1)
                    depths = [m.meas.depth_prior[0] if m.metric else median_depth
                              for m, _ in pairs if isinstance(m, BearingResidual)]
                    has_bearings = bool(depths)
                    R_c, t_c = Rs[u], ts[u]
                    for iteration in range(5 if has_bearings else 1):
                        local, weights = [], []
                        k = 0
                        for m, x in pairs:
                            if isinstance(m, GaussianResidual):
                                local.append(m.mu)
                                weights.append(1.0 / np.trace(np.linalg.pinv(m.P)))
                            else:
                                rho = depths[k] if iteration == 0 else max(np.linalg.norm(x - (R_c @ m.meas.origin + t_c)), 1e-6)
                                k += 1
                                local.append(m.meas.origin + rho * m.meas.direction)
                                weights.append(1.0 / (rho ** 2 * np.trace(m.meas.angular_covariance)))
                        local = np.array(local)
                        w = np.array(weights) / np.sum(weights)
                        cl, cw = w @ local, w @ world
                        if len(pairs) >= 3 and np.linalg.svd(local - cl, compute_uv=False)[1] > 1e-6:
                            U, _, Vt = np.linalg.svd(((local - cl) * w[:, None]).T @ (world - cw))
                            D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
                            R_c = Vt.T @ D @ U.T
                        t_c = cw - R_c @ cl
                    if local_cost(R_c, t_c) < 0.25 * local_cost(Rs[u], ts[u]):
                        Rs[u] = Rs[u] @ _exp_so3(alpha * _log_so3(Rs[u].T @ R_c))
                        ts[u] = ts[u] + alpha * (t_c - ts[u])

    # ---------- one step ----------

    def _cost(self, obs, Rs, ts, required=None):
        c, active = 0.0, set()
        for f, obs_f in obs.items():
            x = self._landmark(obs_f, Rs, ts)
            for i, m in obs_f:
                result = m.evaluate(Rs[i].T @ (x - ts[i]))
                if result is not None:
                    r, _ = result
                    c += 0.5 * r @ r
                    active.add((f, i))
        if required is not None and not required.issubset(active):
            return np.inf, active
        return c, active

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
        self.skipped_bearings = 0
        if not obs:
            self.feature_estimates = {}
            self.pose_covariances = {}
            self.sampled_points = np.zeros((0, 3))
            self.cost = 0.0
            return
        Rs, ts = self._poses()
        comps = self._components(obs)
        # fixed CS per component (it stays where it is): the pinned one if present, else the lowest index
        fixed = {self.pinned_cs_idx if self.pinned_cs_idx in comp else comp[0] for comp in comps}
        for comp in comps:
            if not any(m.metric for rows in obs.values() for i, m in rows if i in comp):
                key = tuple(self.coordinate_systems[i].name for i in comp)
                if key not in self._warned_scale_components:
                    warnings.warn(f"Bearing-only component {key} has no metric information: scale is unobservable.",
                                  RuntimeWarning, stacklevel=2)
                    self._warned_scale_components.add(key)
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
            for i, m in obs[f]:
                v = Rs[i].T @ (x - ts[i])
                result = m.evaluate(v)
                if result is None:
                    self.skipped_bearings += 1
                    continue
                r, D = result
                Jl = D @ Rs[i].T
                Hff += Jl.T @ Jl
                gf += Jl.T @ r
                if i in pose_col:
                    c = pose_col[i]
                    Jp = np.hstack([D @ _skew(v), -Jl])
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

        cost0, active = self._cost(obs, Rs, ts)
        for _ in range(8):  # memoryless backtracking: shrink this step only if it would increase the cost
            Rn, tn = list(Rs), list(ts)
            for i, c in pose_col.items():
                Rn[i] = Rs[i] @ _exp_so3(alpha * delta[c:c + 3])
                tn[i] = ts[i] + alpha * delta[c + 3:c + 6]
            cost1, _ = self._cost(obs, Rn, tn, required=active)
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

        self.sampled_points = np.array([self._landmark(obs[f], Rn, tn) for f in features])

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
