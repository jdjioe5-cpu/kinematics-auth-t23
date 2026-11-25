# `kinematics_auth` — Multi-Channel 2D Kinematics & Burst-Rhythm Anomaly Verifier (`t=23`)

Standalone zero-dependency Python 3.11/3.12 package (`kinematics_auth.py` + `synth_traces.py` + `test_kinematics_auth.py`).

## One-Command End-to-End Run
```bash
python kinematics_auth.py
python -m unittest -v test_kinematics_auth.py
```

## Architecture & Guarantees
- **Hardware-Rate-Agnostic 2D Kinematics**: Uniform arc-length resampling + 3-tap smoothing + stride-3 signed turning angles + normalized velocity centroid & dwell ratio work equally across 110 Hz continuous trajectories and 16 Hz low-velocity streams (verified via 6x hardware downsampling test).
- **Channel-Invariant Burst & Edit Rhythm**: Maps `DIRECT_KEY`, `VIRTUAL_GRID`, and `BATCH_INSERT` (`BATCH_CHUNK`) into dimensionless log-IBI dispersion, lag-1 autocorrelation, skewness, long-pause ratio, edit rate, pre/post-edit hesitation ratios, edit cluster length, and pointer-to-burst handoff coupling.
- **Sequential CUSUM + Shrinkage Mahalanobis**: Ledoit-Wolf style shrinkage covariance (pure stdlib Gauss-Jordan inversion) combined with 4-window sequential CUSUM + worst-window anomaly scoring to detect mid-session takeover and parameter-matched habit mimics.
- **Strict Pre-Holdout Calibration**: Disjoint `enroll` / `calibration` / `holdout` RNG seeds with locked decision threshold (`calibration_locked = True`).

## Verified Results
- **Aggregate FRR**: `0.00%` (`0/36` normal, `0/18` fatigued)
- **Aggregate FAR**: `1.39%` (`1/36` stranger, `0/36` habit mimic)
- **Full `3x3` Swap Matrix**: diagonal `54/54 (100.00%)`, off-diagonal `0/108 (0.00%)`
- **Unit Tests**: `6/6 OK`
