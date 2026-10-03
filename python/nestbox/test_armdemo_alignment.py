# Offline check of the arm demo alignment (docs/ARM_DEMO_INTEGRATION.md, plan step 2).
# Feeds VR/arm calibration pairs to a running daemon as gripper-tip features seen in two coordinate systems,
# pins 'vr', and compares nestbox's roarm->vr estimate against a Kabsch fit of the same pairs.
#
# Needs a fresh daemon (coordinate system names are global) and Redis:
#   redis-server --port 6379 --save "" --appendonly no &
#   PYTHONPATH=<repo>/python python -m nestbox.networking.daemon_server
#   PYTHONPATH=<repo>/python python <repo>/python/nestbox/test_armdemo_alignment.py [--pairs pairs.json]
#
# pairs.json: {"vr": [[x,y,z], ...], "arm": [[x,y,z], ...]}, raw Unity index-tip points and commanded arm
# points in meters. Without --pairs, pairs are simulated from --truth (a follow_hand.py hand_cal.json).
import argparse
import json
import time
import numpy as np
import nestbox
from nestbox.api import _get_client
from nestbox.handedness import mirror_point, mirror_covariance
from nestbox.numutil import coerce_quaternion

CAL_POINTS = np.array([  # follow_hand.py's reachable calibration points, meters, arm frame
    [0.35, -0.10, 0.30],
    [0.25, -0.15, 0.15],
    [0.40, 0.00, 0.10],
    [0.35, 0.10, 0.30],
    [0.28, 0.12, 0.18],
    [0.22, 0.00, 0.28],
])
ARM_SIGMA = 0.005
VR_SIGMA = 0.015

def kabsch(src, dst):
    """R, t minimizing |R @ src + t - dst| (both right-handed)."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs

def simulate_pairs(truth_path, noise, rng):
    cal = json.load(open(truth_path))
    R, t = np.array(cal['R']), np.array(cal['t'])  # follow_hand convention: arm = R @ mirror(vr) + t
    vr_mirrored = (CAL_POINTS - t) @ R  # R.T @ (arm - t), row-wise
    vr_mirrored += rng.normal(scale=noise, size=vr_mirrored.shape)
    vr_unity = np.array([mirror_point(p) for p in vr_mirrored])
    return vr_unity, CAL_POINTS.copy()

def angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', help='recorded pairs json; simulated if omitted')
    ap.add_argument('--truth', default='../arm-demo-20261003/labspace/demo/data/hand_cal.json',
                    help='hand_cal.json used to simulate pairs (path relative to the Nestbox projects folder is typical)')
    ap.add_argument('--noise', type=float, default=0.01, help='simulated per-axis VR noise, meters')
    ap.add_argument('--seconds', type=float, default=60.0, help='how long to watch the optimizer')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.pairs:
        d = json.load(open(args.pairs))
        vr_unity, arm = np.array(d['vr'], dtype=float), np.array(d['arm'], dtype=float)
    else:
        vr_unity, arm = simulate_pairs(args.truth, args.noise, np.random.default_rng(args.seed))
    vr = np.array([mirror_point(p) for p in vr_unity])  # into nestbox's right-handed convention

    # reference: roarm -> vr (mirrored) via Kabsch
    R_ref, t_ref = kabsch(arm, vr)
    print(f"Kabsch roarm->vr residuals (cm): {np.round(np.linalg.norm(arm @ R_ref.T + t_ref - vr, axis=1) * 100, 2)}")

    nestbox.create_coordinate_system('vr')
    nestbox.create_coordinate_system('roarm')
    arm_cov = (np.eye(3) * ARM_SIGMA**2).tolist()
    vr_cov = mirror_covariance(np.eye(3) * VR_SIGMA**2).tolist()
    for k, (p_vr, p_arm) in enumerate(zip(vr, arm)):
        feature = f'nestbox:feature/roarm/gripper-tip/cal/{k}'
        nestbox.add_measurement(feature, 'roarm', p_arm.tolist(), arm_cov)
        nestbox.add_measurement(feature, 'vr', p_vr.tolist(), vr_cov)
    nestbox.pin('vr')
    nestbox.start_alignment()

    client = _get_client()
    start = time.time()
    while time.time() - start < args.seconds:
        time.sleep(5.0)
        result = client.get_transform('roarm', 'vr')  # pose of roarm's frame in vr: p_vr = R(q) p_roarm + origin
        R = coerce_quaternion(result.quaternion).rotation_matrix
        o = np.array(result.origin)
        rms = np.sqrt(np.mean(np.sum((arm @ R.T + o - vr) ** 2, axis=1)))
        print(f"t={time.time() - start:5.1f}s  vs Kabsch: {np.linalg.norm(o - t_ref) * 100:6.2f} cm, "
              f"{angle_deg(R, R_ref):6.2f} deg   fit RMS {rms * 100:5.2f} cm   q(wxyz)={np.round(result.quaternion, 3)}")

    # the public conversion path should agree with the raw transform
    converted = np.array(nestbox.from_cs('roarm').to_cs('vr').convert_many(arm.tolist()))
    print(f"convert() vs Kabsch prediction (cm): {np.round(np.linalg.norm(converted - (arm @ R_ref.T + t_ref), axis=1) * 100, 2)}")

if __name__ == '__main__':
    main()
