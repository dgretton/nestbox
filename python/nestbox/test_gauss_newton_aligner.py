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


# Bearing checks use their own seeded RNG so the eight Gaussian regression printouts stay unchanged.
from nestbox.measurement import BearingMeasurement
from nestbox.aligner.measurement_residual import BearingResidual, GaussianResidual
from nestbox.aligner.gauss_newton_aligner import _matrix_to_quat, _skew
import warnings

BEARING_POINTS = np.array([[-.5, -.3, 1], [.4, -.3, 1.2], [.5, .4, 1.5], [-.4, .4, 1.8],
                           [.1, .1, 2], [-.2, -.1, 2.3], [.6, .2, 2.5], [-.6, .2, 2.2]])


def make_bearing_aligner(noise=False, mixed=False, seed=0):
    random = np.random.default_rng(seed)
    al, css = make_aligner(names=('vr', 'arm', 'cam') if mixed else ('vr', 'cam'), noise=False,
                          features=BEARING_POINTS)
    R, t = TRUTH['cam']
    for k, x in enumerate(BEARING_POINTS):
        f = to_feature(f'f{k}')
        vr_cov = np.eye(3) * .002 ** 2
        css['vr'].measurements[f] = NormalMeasurement(f, x + (random.normal(size=3) * .002 if noise else 0), vr_cov)
        # A nonzero sensor origin and correlated anisotropic noise exercise the full model.
        origin = np.array([.08, -.03, .02])
        d = R.T @ (x - t) - origin
        d /= np.linalg.norm(d)
        basis = BearingMeasurement.tangent_basis(d)
        C = np.array([[4e-6, 1e-6], [1e-6, 2e-6]])
        observed = d + (basis @ random.multivariate_normal(np.zeros(2), C) if noise else 0)
        observed /= np.linalg.norm(observed)
        # Project the old first image axis to the new tangent plane.
        e = basis[:, 0] - observed * (observed @ basis[:, 0])
        e /= np.linalg.norm(e)
        E = np.column_stack((e, np.cross(observed, e)))
        css['cam'].measurements[f] = BearingMeasurement(f, observed, C, basis=E, origin=origin)
        if mixed:
            mu, cov = measure('arm', x, noise=False)
            if noise:
                mu += random.multivariate_normal(np.zeros(3), cov)
            css['arm'].measurements[f] = NormalMeasurement(f, mu, cov)
    return al, css


def test_bearing_jacobians():
    m = BearingMeasurement(to_feature('jacobian'), [.2, -.1, 1], [[.01, .002], [.002, .02]],
                           origin=[.1, -.2, .3], depth_prior=(2., .8))
    model = BearingResidual(m)
    v = np.array([.5, .1, 2.])
    r, J = model.evaluate(v)
    h = 1e-6
    numeric = np.column_stack([(model.evaluate(v + h * e)[0] - model.evaluate(v - h * e)[0]) / (2*h)
                               for e in np.eye(3)])
    assert np.allclose(J, numeric, atol=1e-8)
    # Chain rule for the aligner's body rotation and world translation convention.
    R = _exp_so3(np.array([.3, -.1, .2]))
    numeric_rotation = np.column_stack([(model.evaluate(_exp_so3(-h*e) @ v)[0] -
                                        model.evaluate(_exp_so3(h*e) @ v)[0])/(2*h) for e in np.eye(3)])
    assert np.allclose(J @ _skew(v), numeric_rotation, atol=1e-8)
    assert model.evaluate(m.origin - m.direction) is None
    print('bearing residual and pose Jacobians: finite differences agree')


def test_bearing_pnp_exact():
    al, css = make_bearing_aligner()
    run(al, 20)
    d, a = pose_error(al, css['cam'], 'cam')
    assert d < 1e-6 and a < 1e-4, (d, a)
    assert len(al.measurement_cones()) == len(BEARING_POINTS)
    assert len(al.measurement_ellipsoids()) == len(BEARING_POINTS)
    print(f'bearing PnP from identity: position error {d:.1e} m, angle {a:.1e} deg')


def test_bearing_covariance_matches_scatter():
    for mixed in (False, True):
        est, pred = [], []
        for seed in range(300):
            al, css = make_bearing_aligner(noise=True, mixed=mixed, seed=seed)
            run(al, 15)
            est.append(pose(al, css['cam'])[1])
            pred.append(al.pose_covariances[css['cam'].name][3:, 3:])
        ratio = np.trace(np.cov(np.array(est).T)) / np.trace(np.mean(pred, axis=0))
        bias = np.linalg.norm(np.mean(est, axis=0) - TRUTH['cam'][1])
        assert .75 < ratio < 1.33, (mixed, ratio)
        assert bias < .003, (mixed, bias)
        print(f'bearing camera covariance (mixed={mixed}, 300 trials): empirical/predicted {ratio:.2f}', flush=True)


def test_bearing_cheirality():
    al, css = make_bearing_aligner()
    run(al, 20)
    R, t = TRUTH['cam']
    f = to_feature('behind')
    css['vr'].measurements[f] = NormalMeasurement(f, t + R @ np.array([0, 0, -2]), np.eye(3)*1e-6)
    css['cam'].measurements[f] = BearingMeasurement(f, [0, 0, 1], .002)
    run(al, 10)
    assert al.skipped_bearings == 1
    d, a = pose_error(al, css['cam'], 'cam')
    assert d < 1e-6 and a < 1e-4, (d, a)
    print('cheirality: behind-camera bearing skipped without pose bias')


def test_bearing_scale_gauge():
    al = GaussNewtonAligner(convergence_time=0)
    css = [CoordinateSystem(f'scale_{next(_names)}') for _ in range(2)]
    centers = [np.zeros(3), np.array([.3, .1, 0])]
    for cs, center in zip(css, centers):
        for k, x in enumerate(BEARING_POINTS):
            f = to_feature(f'f{k}')
            cs.measurements[f] = BearingMeasurement(f, x-center, .002)
        al.add_coordinate_system(cs, initial_origin=center)
    al.pin(css[0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        run(al, 10)
        before = pose(al, css[1])[1].copy()
        run(al, 200)
    drift = np.max(np.abs(pose(al, css[1])[1] - before))
    axis = centers[1] / np.linalg.norm(centers[1])
    variance = axis @ al.pose_covariances[css[1].name][3:, 3:] @ axis
    assert len(caught) == 1 and 'scale is unobservable' in str(caught[0].message), caught
    assert drift < 1e-7 and variance > 1e3, (drift, variance)
    assert np.all(np.isfinite(al.sampled_points))
    print(f'bearing-only scale gauge: drift {drift:.1e}, scale variance {variance:.1e}, one warning')


def test_bearing_far_points():
    # Angular information is independent of range. A fixed-d0 Gaussian matches at d0,
    # but discards angular information for nearer features if d0 was chosen for the sky.
    m = BearingMeasurement(to_feature('far'), [.2, -.1, 1], .002)
    b = BearingResidual(m)
    d = m.direction
    Jnear = b.evaluate(d)[1] @ _skew(d)
    Jfar = b.evaluate(1000*d)[1] @ _skew(1000*d)
    assert np.allclose(Jnear.T @ Jnear, Jfar.T @ Jfar)
    C = (1000*.002)**2 * (np.eye(3)-np.outer(d,d)) + 1e8*np.outer(d,d)
    g = GaussianResidual(1000*d, C)
    Jg = g.W @ _skew(d)
    ratio = np.trace(Jnear.T @ Jnear) / np.trace(Jg.T @ Jg)
    assert np.isclose(ratio, 1e6)
    assert np.allclose((1000*Jg).T @ (1000*Jg), Jfar.T @ Jfar)
    # Solve an actual kilometer-scale scene and inspect its rotation covariance.
    al, css = make_bearing_aligner()
    for k, x in enumerate(1000 * BEARING_POINTS):
        f = to_feature(f'f{k}')
        css['vr'].measurements[f] = NormalMeasurement(f, x, np.eye(3)*.002**2)
        R, t = TRUTH['cam']
        css['cam'].measurements[f] = BearingMeasurement(f, R.T @ (x-t), .002)
    run(al, 30)
    dpos, angle = pose_error(al, css['cam'], 'cam')
    rot_sd = np.sqrt(np.max(np.linalg.eigvalsh(al.pose_covariances[css['cam'].name][:3, :3])))
    assert dpos < 1e-5 and angle < 1e-4 and rot_sd < .02, (dpos, angle, rot_sd)
    print(f'1 km features: rotation sd {np.degrees(rot_sd):.2f} deg; fixed-1km Gaussian loses {ratio:.0f}x information at 1m')


def test_bearing_depth_prior():
    f = to_feature('depth')
    gaussian = GaussianResidual(np.array([0., 0., 10.]), np.eye(3))
    values = []
    for sigma in (100., .01):
        bearing = BearingResidual(BearingMeasurement(f, [0, 0, 1], .002, depth_prior=(2., sigma)))
        values.append(GaussNewtonAligner._landmark([(0, gaussian), (1, bearing)], [np.eye(3)]*2, [np.zeros(3)]*2)[2])
    assert abs(values[0]-10) < .001 and abs(values[1]-2) < .01, values
    print(f'log-depth prior: weak -> {values[0]:.3f} m; strong -> {values[1]:.3f} m (prior 2m, measured 10m)')


def test_bearing_fixed_point_and_restart():
    al, css = make_bearing_aligner(noise=True, mixed=True, seed=123)
    run(al, 30)
    before = np.array(al.current_origins)
    rotations_before = np.array(al.current_orientations)
    run(al, 200)
    drift = max(np.max(np.abs(before-al.current_origins)),
                np.max(np.abs(rotations_before-al.current_orientations)))
    saved = (list(al.current_origins), list(al.current_orientations))
    fresh = GaussNewtonAligner(convergence_time=0)
    for i, cs in enumerate(al.coordinate_systems):
        fresh.add_coordinate_system(cs, saved[0][i], saved[1][i])
    fresh.pin(css['vr'])
    run(fresh, 1)
    jump = max(np.max(np.abs(np.array(saved[0])-fresh.current_origins)),
               np.max(np.abs(np.array(saved[1])-fresh.current_orientations)))
    assert drift < 1e-9 and jump < 1e-9, (drift, jump)
    print(f'bearings fixed-point drift {drift:.1e}; restart jump {jump:.1e}')


if __name__ == '__main__':
    for test in [test_noise_free_exact, test_still_at_fixed_point, test_covariance_matches_scatter, test_restart_is_seamless,
                 test_follows_data_change, test_weak_geometry_stays_put, test_new_cs_with_large_rotation, test_camera_rays,
                 test_bearing_jacobians, test_bearing_pnp_exact, test_bearing_cheirality, test_bearing_scale_gauge,
                 test_bearing_far_points, test_bearing_depth_prior, test_bearing_fixed_point_and_restart,
                 test_bearing_covariance_matches_scatter]:
        test()
    print("all passed")
