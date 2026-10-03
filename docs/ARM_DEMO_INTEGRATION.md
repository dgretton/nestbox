# Old nestbox × Quest/RoArm demo: modernization and integration notes

Written 2026-10-03, by the session that built the hand-follow arm demo, for the agent picking this up.
Everything marked **verified** was run on this Mac (macOS, Python 3.13.3) that day. Everything else is
reading of the code, and should be checked.

## Status update (2026-10-03, second session, branch `armdemo`)

`armdemo` = `main` + `torchgpu`, plus uncommitted fixes. All of the following was **verified** in a fresh venv
(`pip install -r python/requirements-armdemo.txt`):

- **Plan step 1 done:** with the Unix socket, `test_toplevel_setup.py` gives cs1→cs2 `[2,3,4]` → `[-2.012, 0.981, 2.999]`.
- **Plan step 2 done on simulated pairs:** run `python/nestbox/test_armdemo_alignment.py`. It simulates the 6 `CAL_POINTS` from
  the lab `hand_cal.json` with 1 cm noise, mirrors them, pins `vr`, and compares against Kabsch. Starting from identity (no
  Kabsch seed needed, see gap 9), it gets to within 0.4 cm / 0.3° of Kabsch in about 15 s. After that it jitters by
  about 0.3–2 cm / 0.3–3°, because the aligner samples stochastically, so the Unity box should lerp. Use `--pairs` once real raw pairs are recorded.
- **Torchgpu regression:** after the merge, `datatype: float32` auto-selected **MPS**, and the aligner thread died on its first step
  (`can't convert mps:0 device type tensor to numpy`). The model mixes CPU tensors and `.numpy()` calls with the device. Device
  selection is now an aligner config key, `device`, default `cpu` (`auto` restores the old behavior).
- **Daemon wedge:** any request that raised inside the aligner server (e.g. a transform for an unknown CS name) killed its only
  connection thread, and every later request hung. Errors now come back as a 500, and the daemon keeps serving.
- Gap 1: `nestbox/handedness.py` has `mirror_point`, `mirror_covariance`, `mirror_quaternion` and `mirror_pose`.
- Gap 2: the shim names features `hand/<stream>/bone/<id>/position` (all 24), mirrors x and the covariance, reads the 4-byte length
  robustly, and no longer prints per message. `config/armdemo_config.yaml` points it at `127.0.0.1:12345`.
- Gap 5: the pin bug is fixed. Pinning by name didn't work anyway, because the aligner only knows CS GUIDs and the `pin` request was
  a no-op. `nestbox.pin('vr')` now goes through a new route, `PUT /coordsys/<cs>/pin`.
- Gap 6: **answered.** `GET /transforms/convert/roarm/to/vr` returns the pose of roarm's frame in (right-handed) vr: `p_vr = R(q)·p_roarm + origin`, with
  `quaternion` in **w, x, y, z** order. For Unity: position `(-ox, oy, oz)`, and rotation `new Quaternion(qx, -qy, -qz, qw)`
  (the mirror, plus Unity's x,y,z,w constructor order). `config.py` now honors `NESTBOX_CONFIG`, `--config-path` drives the daemon's connection,
  and the Python client speaks plain HTTP when `server_type: tcp`. `config/armdemo_config.yaml` serves on `127.0.0.1:8650`, verified with curl
  and the test script. Use `adb reverse tcp:8650 tcp:8650`.
- Gap 10: the `~` makedirs bug is fixed. Per-step and per-request prints and werkzeug access logs are quiet. Versions are pinned in `requirements-armdemo.txt`.

**Still open:** plan steps 3–5, which are `follow_hand.py --nestbox` (outside this repo), the Unity `NestboxCoordinateBox`, and the
history-ring re-drift. Also open: recording real raw calibration pairs, the port-12345 ownership choice (gap 3), and the GPU path.

## Goal

Use this (old) nestbox package as a working baseline for an undergrad researcher's integration work,
and show in the Quest a **box representing the robot arm's coordinate system** that:

1. **drifts** to the pose most consistent with the current measurements (i.e. watch the optimizer converge), and
2. **re-drifts** to a new pose when the arm is physically moved. That needs direct tracking of the arm, which
   is in the works and not available yet.

This is not `nestbox-ng` (the future rewrite). Keep changes pragmatic and local.

## What exists today (outside this repo)

`~/Dropbox (Personal)/Personal Projects/Nestbox/arm-demo-20261003/` (read its README):

- **Quest app** (`apk/nestboxhand-armdemo.apk`, built from `../NestboxHand` with overlay files in `unity/`).
  It streams both hands as length-prefixed `Twig` protobufs over TCP to `127.0.0.1:12345`, carried to
  the Mac by `adb reverse tcp:12345 tcp:12345`, at about 9 Hz per hand. Each Twig is **24 points = raw
  `OVRSkeleton.Bones` positions** in Unity world space (OVR bone IDs: 0 WristRoot, 1 ForearmStub, 2–5 Thumb0–3,
  6–8 Index1–3, 9–11 Middle1–3, 12–14 Ring1–3, 15–18 Pinky0–3, 19 ThumbTip, 20 IndexTip, 21–23 Middle/Ring/PinkyTip).
  `streamId` is `lefthand`/`righthand`, `coordSysId` is `base`. Passthrough is on. It holds 72 fps.
- **`labspace/demo/follow_hand.py`**: receives the hands, does a **standalone Kabsch fit** (not this
  package's aligner) from 6 touch-the-gripper calibration points, then makes the arm follow your pinch. Arm
  control goes through the student's `labspace/roarm/bridge.py` (SSH tunnel) or directly to `roarm_server.py`
  on the Pi (`--arm-host 100.78.66.117`; JSON over TCP on port 8659, meters, about 1.2 s per move).
- First lab run: a 4-point Kabsch fit gave 1.3–2.9 cm residuals (about 2.2 cm RMS). That's a realistic noise
  floor for fingertip-touches-gripper pairs.

**The deliverable is to replace that Kabsch fit with this package's aligner, and show its live estimate in VR.**

## Baseline status: the aligner works on modern Python (verified)

In a fresh venv (`python3.13 -m venv`, `pip install -r python/requirements.txt`), the requirements resolved to
torch 2.14.1, numpy 2.5.3, protobuf 7.36.2, Flask 2.2.5, Werkzeug 3.1.9, Flask-RESTful 0.3.10,
redis(-py) 8.1.0 and pyquaternion 0.9.9. Then:

```
redis-server --port 6379 --save "" --appendonly no &          # aligner thread publishes to Redis
cd <somewhere outside the repo>                                  # see the "~" bug below
PYTHONPATH=<repo>/python MPLBACKEND=Agg python -m nestbox.networking.daemon_server
PYTHONPATH=<repo>/python python <repo>/python/nestbox/test_toplevel_setup.py
python -c "import nestbox; print(nestbox.from_cs('cs1').to_cs('cs2').convert([2,3,4]))"
```

- Every module imports.
- The daemon serves Flask on `~/.nestbox/socket/daemon.sock`.
- The Adam aligner converges: cs1→cs2 conversions land within about 1–2 cm of the expected points (`[2,3,4]` → `[-2.003, 0.993, 3.016]`, expected `[-2, 1, 3]`).
- A Quest-shaped Twig (24 samples plus a 3×3 identity transform) parses under protobuf 7 with the 5.26-generated `twig_pb2.py`.
- The old `python/venv` is dead: it points at pyenv Python 3.7.3, which no longer exists. Make a new one, and pin the versions above (`pip freeze`) once things work.

## Branches

`main` (2025-01-13, "metrology module… arm-to-dock demo") is the base.

- `torchgpu`: +2, includes `alignerconfig`. Passes the aligner config yaml through to the aligner, and adds dtype plus `mps`/`cuda` selection. Probably worth merging.
- `poseinterp`: +10, on GitHub. Evan's pose interpolation from calibration data; its own commits call it "messy". Optional.
- `sourcemanager`: currently checked out, +16. Source-concatenation tooling only; unrelated.
- Untracked: `python/nestbox/fastapi_daemon_server.py` (someone started a FastAPI port), `csharp/Nestbox.Core/Core.cs`, `TestConnection.cs`, and `python/vis_mvg_pdf_ray_integral/`. Ask Dana before deleting or committing any of these.

Suggested: branch `armdemo` off `main`, merge `torchgpu`, and work there.

## Gaps, in the order they'll bite

1. **Handedness (P0).**
   - The problem: Unity is left-handed (y up). The arm and nestbox's quaternion math are right-handed. No proper rotation maps one onto the other, so the aligner will fit garbage or stall. `follow_hand.py` and the old `labspace/demo/move_arm_to_dock.py` both mirror Unity x before fitting.
   - Do the same at nestbox's ingestion boundary: negate x of every VR point.
   - Covariances: for diagonal ones, nothing changes; otherwise apply S·Σ·S with S = diag(−1, 1, 1).
   - When sending a pose back to Unity, mirror again: position x → −x, quaternion (w, x, y, z) → (w, x, −y, −z).

2. **`live_data_shim.py` doesn't match the Quest stream (verified).**
   - It labels points as `hand/<stream>/root-pose/position` plus `trackingpoint/0..19` and zips that against the samples. 24 bones become **21 features**: bones 0–20 get mislabeled, and the middle, ring and pinky tips are dropped. The index tip comes out named `trackingpoint/19`.
   - Name features by OVR bone ID instead, e.g. `nestbox:feature/hand/righthand/bone/20/position`.
   - The shim also only starts with `--live`. It binds to the host in `peer_discovery.temp_live_data_shim_address` (`10.1.20.59:12345`, not an address on this Mac; use `127.0.0.1` or `0.0.0.0`). And it prints per message.

3. **Port 12345 is contested.** `follow_hand.py` and the shim both want it. Pick one owner:
   - **Simplest:** keep `follow_hand.py` as the hand receiver and arm driver, and have it *push* measurements to the daemon with `nestbox.add_measurements(...)`.
   - Or let the daemon own the stream, and have `follow_hand.py` read the aligned transform from the daemon.

4. **Arm measurements: model the gripper tip as features observed in two coordinate systems.**
   - Create coordinate systems `vr` and `roarm`, and **pin `vr`**.
   - For calibration touch k, add feature `nestbox:feature/roarm/gripper-tip/cal/<k>`:
     - in `roarm`: the commanded arm position (meters), σ ≈ 0.5 cm;
     - in `vr`: the mirrored index-tip position (bone 20), averaged over about 0.75 s, σ ≈ 1.5 cm. The ~2 cm Kabsch residuals above are the evidence for that σ.
   - Three or more non-collinear touches constrain the pose. The 6-point list in `follow_hand.py` (`CAL_POINTS`) is known to be reachable, and the first lab run reached its far points.

5. **Pinning bug.** In `aligner.Aligner.pin`, the `CoordinateSystem` branch computes an index but never assigns `self.pinned_cs_idx`. Pin by name (`'vr'`) or by int, or fix that one line.

6. **Getting the estimate into Unity: the C# side can't receive.**
   - `csharp/Nestbox.Interfaces/IConnection` only has `Send`, so there's no reverse channel.
   - Recommended: run the daemon's Flask server over TCP instead of the Unix socket, and have Unity poll `GET /transforms/convert/roarm/to/vr` at about 10 Hz with `UnityWebRequest`. Add `adb reverse tcp:<port> tcp:<port>`.
   - The response is `AlignmentResult.to_json()`: `{timestamp, status, origin, quaternion, delta_velocity}`. **Verify the quaternion order** (pyquaternion is w-first; the daemon log prints orientations like `[0.78, -0.43, -0.14, -0.42]`) and whether origin/quaternion map roarm→vr or the reverse. `api.py`'s `OptimizedTransformer` shows how the Python client reads it. Test by converting the 6 calibration points.
   - Catch: `nestbox/config.py` reads the **daemon** connection (`server_type`/`server_address`) from `config/example_config.yaml` at import time, ignoring `--config-path`. Switching to TCP means editing that file or fixing `config.py`.

7. **Unity box.**
   - A new MonoBehaviour in `NestboxHand` (e.g. `NestboxCoordinateBox`) polls the endpoint, mirrors back into Unity handedness, and moves a wireframe cube plus axis gizmo at the arm base, along with a marker at the predicted gripper tip.
   - Lerp toward each new pose: the "drift" Dana wants is mostly the optimizer itself (`run_optimizer` steps every 0.066 s).
   - **Keep its per-frame cost tiny.** Today's arm-demo build ran at 13–52 fps until the debug panel and per-message `Debug.Log` were fixed, so log only on state changes.
   - Build with the menu Nestbox → Build Arm Demo APK, from a copy of `NestboxHand` outside Dropbox; see the arm-demo README for the overlay files and the Unity licensing/adb gotchas.

8. **Re-drift when the arm moves (needs the tracking that's coming).**
   - Measurements are stored per feature and replaced on update. A *single* moving gripper-tip feature only ever gives one point pair, which isn't enough to constrain rotation.
   - Use a ring buffer of the last N (≈ 8–12) arm/VR pairs as distinct features, `.../gripper-tip/hist/<i mod N>`, so when the arm base moves, old pairs get overwritten and the box drifts to the new pose.
   - Better later: timestamp the pairs and inflate covariance with age.
   - `AlignerManager` already has per-stream clear keys (`_clear_measurements` / `_update_clear_keys`) that may help.

9. **Initialization.** The gradient aligner starts from whatever `add_coordinate_system` gives it, and VR versus arm frames differ by large rotations. If convergence is slow or lands wrong, seed `roarm`'s pose with a Kabsch fit (`fit_rigid` in `follow_hand.py`) via `Aligner.reset_coordinate_system`. Then let the optimizer refine and visibly drift from there.

10. **Smaller things.**
    - `daemon_server.py`, when the socket directory is missing, runs `os.makedirs("~/.nestbox/socket")` with a literal `~`, which creates a folder named `~` in the current directory. Run from outside the repo, or fix it with `expanduser`.
    - `@app.before_first_request` is removed in Flask 2.3+ (requirements pin `<2.3`, fine for now).
    - The aligner thread prints every coordinate system every step, and the daemon logs every queue receive. Quiet both before relying on the console.
    - `requirements.txt` is unpinned; pin the tested set.

## Suggested plan, with acceptance checks

1. **Environment.** New venv, temporary Redis, daemon up, and `test_toplevel_setup.py` converging as above. *Check:* cs1→cs2 within 2 cm.
2. **Offline alignment.** A script that takes recorded calibration pairs (`labspace/demo/data/hand_cal.json` only stores the Kabsch result; record raw pairs from `follow_hand.py`'s calibration step) and feeds them to the daemon as in gap 4, with handedness fixed. *Check:* roarm→vr from nestbox matches the Kabsch transform to about 1 cm and 1–2°.
3. **Live alignment.** `follow_hand.py --nestbox` (new flag) pushes each calibration pair to the daemon and, in follow mode, converts targets with `nestbox.from_cs('vr').to_cs('roarm')` instead of the Kabsch matrix. *Check:* the arm follows as well as with Kabsch.
4. **Box in VR.** TCP daemon endpoint, adb reverse, and the Unity box. *Check:* the box starts from the initial guess, drifts, and settles on the arm base, steady at 72 fps.
5. **Re-drift.** The history-ring features (gap 8), first fed by a *simulated* arm move: shift the arm-side positions by a known offset. *Check:* the box drifts by that offset. Then wire in real arm tracking when it exists.

## Pointers

- `python/nestbox/daemon.py` (singleton `global_daemon`), `daemon_local_aligner_client.py` (aligner thread, Redis, visualizer), `aligner/torch_aligner.py` (`AdamAligner`), `api.py` (top-level `nestbox.*`), `networking/daemon_server.py` (Flask routes), `live_data_shim.py`.
- Arm demo: `../arm-demo-20261003/README.md`, `labspace/demo/FOLLOW_HAND.md`. Arm server protocol: `labspace/roarm/roarm_server.py` (`move_xyzt` with x/y/z in meters).
- Measure Quest frame rate with `adb logcat -s VrApi` (lines like `FPS=72/72`).
- Running Unity on this Mac resets the Quest's adb authorization; re-accept "Allow USB debugging" in the headset.
