# Checks for GaussNewtonAligner (no daemon needed): python -m nestbox.test_gauss_newton_aligner
# Scene: 'vr' (pinned), 'arm' (isotropic ~5 mm), 'cam' (camera-like: 2 mm across its view rays, 0.5 m along them).
import itertools
import numpy as np
from nestbox.aligner.gauss_newton_aligner import GaussNewtonAligner, _exp_so3, _quat_to_matrix
from nestbox.coordsystem import CoordinateSystem
from nestbox.measurement import NormalMeasurement
from nestbox.feature import to_feature

_names = itertools.count()
rng = np.random.default_rng(0)

FEATURES = np.array([[0.4, -0.1, 1.0], [0.3, 0.2, 1.1], [0.5, 0.0, 0.8], [0.35, 0.15, 1.3], [0.25, -0.2, 0.9], [0.45, 0.1, 1.2]])
TRUTH = {  # pose of each CS in vr: p_vr = R p_local + t
    'vr': (np.eye(3), np.zeros(3)),
    'arm': (_exp_so3(np.array([0.3, -1.2, 2.0])), np.array([0.6, -0.4, 0.2])),
    'cam': (_exp_so3(np.array([-0.4, 2.6, 0.1])), np.array([-0.5, 0.3, 1.0])),
}
SIGMA = {'vr': 0.015, 'arm': 0.005}


def local_cov(cs, p_local):
    if cs == 'cam':  # elongated along the view ray from the camera origin
        d = p_local / np.linalg.norm(p_local)
        return 0.002 ** 2 * (np.eye(3) - np.outer(d, d)) + 0.5 ** 2 * np.outer(d, d)
    return np.eye(3) * SIGMA[cs] ** 2


def measure(cs, x_world, noise=True):
    R, t = TRUTH[cs]
    p = R.T @ (x_world - t)
    C = local_cov(cs, p)
    return (rng.multivariate_normal(p, C) if noise else p), C


def make_aligner(names=('vr', 'arm', 'cam'), noise=True, features=FEATURES, **kw):
    al = GaussNewtonAligner(convergence_time=0, **kw)
    css = {}
    for n in names:
        cs = CoordinateSystem(f'{n}_{next(_names)}')
        for k, x in enumerate(features):
            mu, C = measure(n, x, noise)
            cs.measurements[to_feature(f'f{k}')] = NormalMeasurement(to_feature(f'f{k}'), mu, C)
        al.add_coordinate_system(cs)
        css[n] = cs
    al.pin(css['vr'])
    return al, css


def pose(al, cs):
    i = al.coordinate_systems.index(cs)
    return _quat_to_matrix(al.current_orientations[i]), np.array(al.current_origins[i])


def pose_error(al, cs, name):
    R, t = pose(al, cs)
    Rt, tt = TRUTH[name]
    ang = np.degrees(np.arccos(np.clip((np.trace(R.T @ Rt) - 1) / 2, -1, 1)))
    return np.linalg.norm(t - tt), ang


def chi2(al, cs, name):
    """Error vs truth, in the aligner's own 6-dof parameterization, weighted by its reported covariance."""
    R, t = pose(al, cs)
    Rt, tt = TRUTH[name]
    dR = R.T @ Rt  # R_true = R_est Exp(phi)
    ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
    phi = np.zeros(3) if ang < 1e-12 else ang / (2 * np.sin(ang)) * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
    e = np.concatenate([phi, tt - t])
    return float(e @ np.linalg.solve(al.pose_covariances[cs.name], e))


def run(al, steps):
    for _ in range(steps):
        al.gradient_descent_step()


def test_noise_free_exact():
    al, css = make_aligner(noise=False)
    run(al, 30)
    for n in ('arm', 'cam'):
        d, a = pose_error(al, css[n], n)
        assert d < 1e-6 and a < 1e-4, (n, d, a)
    print(f"noise-free: exact recovery, cost {al.cost:.2e}")


def test_still_at_fixed_point():
    al, css = make_aligner()
    run(al, 30)
    before = [pose(al, css[n]) for n in ('arm', 'cam')]
    run(al, 200)
    after = [pose(al, css[n]) for n in ('arm', 'cam')]
    drift = max(max(np.abs(a[0] - b[0]).max(), np.abs(a[1] - b[1]).max()) for a, b in zip(before, after))
    assert drift < 1e-9, drift
    print(f"static data: max change over 200 more steps {drift:.1e} (no jitter)")


def test_covariance_matches_scatter():
    est, pred = [], []
    for _ in range(300):
        al, css = make_aligner()
        run(al, 15)
        est.append(pose(al, css['arm'])[1])
        pred.append(al.pose_covariances[css['arm'].name][3:, 3:])
    emp = np.cov(np.array(est).T)
    ratio = np.trace(emp) / np.trace(np.mean(pred, axis=0))
    assert 0.75 < ratio < 1.33, ratio
    print(f"arm translation: empirical/predicted variance {ratio:.2f} (Laplace covariance is honest); "
          f"predicted sd {np.sqrt(np.trace(np.mean(pred, axis=0)) / 3) * 100:.2f} cm")


def test_restart_is_seamless():
    al, css = make_aligner()
    run(al, 30)
    saved = (list(al.current_origins), list(al.current_orientations))
    fresh = GaussNewtonAligner(convergence_time=0)
    for i, cs in enumerate(al.coordinate_systems):
        fresh.add_coordinate_system(cs, saved[0][i], saved[1][i])
    fresh.pin(css['vr'])
    fresh.gradient_descent_step()
    jump = max(np.abs(np.array(a) - np.array(b)).max() for a, b in zip(saved[0], fresh.current_origins))
    assert jump < 1e-9, jump
    print(f"restart from saved estimate: first-step jump {jump:.1e}")


def test_follows_data_change():
    al, css = make_aligner(names=('vr', 'arm'), noise=False)
    run(al, 30)
    R, t = TRUTH['arm']
    shift = np.array([0.1, 0.0, -0.05])  # the arm base was moved; its own measurements are unchanged, VR sees it moved
    for k, x in enumerate(FEATURES):
        mu, C = measure('vr', x + shift, noise=False)
        css['vr'].measurements[to_feature(f'f{k}')] = NormalMeasurement(to_feature(f'f{k}'), mu, C)
    run(al, 30)
    _, t_new = pose(al, css['arm'])
    assert np.linalg.norm(t_new - (t + shift)) < 1e-6, t_new - (t + shift)
    print("data change: re-converged to the moved pose")


def test_weak_geometry_stays_put():
    al, css = make_aligner(names=('vr', 'arm'), features=FEATURES[:2])  # 2 shared points: rotation about their line is free
    run(al, 20)
    before = pose(al, css['arm'])
    run(al, 300)
    after = pose(al, css['arm'])
    drift = max(np.abs(after[0] - before[0]).max(), np.abs(after[1] - before[1]).max())
    big = np.max(np.linalg.eigvalsh(al.pose_covariances[css['arm'].name]))
    assert drift < 1e-7 and big > 1e3, (drift, big)  # sub-micron: floating-point dust, not a random walk
    print(f"2-point (underdetermined) geometry: drift {drift:.1e}; free rotation reported with variance {big:.1e}")


def test_new_cs_with_large_rotation():
    al, css = make_aligner(names=('vr', 'arm'))
    run(al, 10)
    cs = CoordinateSystem(f'late_{next(_names)}')
    TRUTH['late'] = (_exp_so3(np.array([0.0, 0.0, np.radians(175)])), np.array([1.0, 1.0, 0.0]))
    SIGMA['late'] = 0.005
    for k, x in enumerate(FEATURES):
        mu, C = measure('late', x)
        cs.measurements[to_feature(f'f{k}')] = NormalMeasurement(to_feature(f'f{k}'), mu, C)
    al.add_coordinate_system(cs)  # starts at the identity, 175 degrees from the truth
    run(al, 10)
    d, a = pose_error(al, cs, 'late')
    c2 = chi2(al, cs, 'late')
    assert a < 20 and c2 < 22.5, (d, a, c2)  # placed (no flip), error consistent with its covariance (chi2_6 p=0.999)
    print(f"coordinate system appearing 175 deg off: placed, now {d * 100:.1f} cm / {a:.1f} deg from truth, "
          f"chi2 vs its own covariance {c2:.1f} (6 dof)")


def test_camera_rays():
    al, css = make_aligner()
    run(al, 30)
    d, a = pose_error(al, css['cam'], 'cam')
    c2 = chi2(al, css['cam'], 'cam')
    assert c2 < 22.5, (d, a, c2)
    print(f"camera-like CS (2 mm x 0.5 m ray covariances): {d * 100:.1f} cm / {a:.2f} deg from truth, chi2 {c2:.1f} (6 dof)")


if __name__ == '__main__':
    for test in [test_noise_free_exact, test_still_at_fixed_point, test_covariance_matches_scatter, test_restart_is_seamless,
                 test_follows_data_change, test_weak_geometry_stays_put, test_new_cs_with_large_rotation, test_camera_rays]:
        test()
    print("all passed")
