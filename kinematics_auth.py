"""kinematics_auth.py — Zero-dependency Python 3.12 multi-channel 2D kinematics
and burst-rhythm anomaly verifier (t=23).

Strictly stdlib-only (no third-party packages).
Provides:
  - Hardware-rate-agnostic 2D trajectory arc-length resampling & geometric/dynamic invariants
  - Channel-invariant symbol-arrival & edit/revision rhythm invariants across
    (DIRECT_KEY, VIRTUAL_GRID, BATCH_INSERT)
  - Regularized covariance (Mahalanobis) + sequential windowed CUSUM scoring
  - Strict pre-holdout threshold calibration (zero test-set leakage)
  - One-command CLI benchmark runner (delegating to synth_traces.py)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

CANONICAL_CHANNELS = ("DIRECT_KEY", "VIRTUAL_GRID", "BATCH_INSERT")
_CHANNEL_ALIASES = {
    "DIRECT_KEY": "DIRECT_KEY",
    "VIRTUAL_GRID": "VIRTUAL_GRID",
    "BATCH_INSERT": "BATCH_INSERT",
    "BATCH_CHUNK": "BATCH_INSERT",
}


def normalize_channel(channel: str) -> str:
    key = channel.strip().upper()
    if key not in _CHANNEL_ALIASES:
        raise ValueError(f"Unsupported symbol burst channel: {channel!r}")
    return _CHANNEL_ALIASES[key]


@dataclass(frozen=True)
class TrajectoryTick:
    """Normalized 2D pointer/cursor trajectory sample."""
    ts_ms: float
    x_norm: float
    y_norm: float
    act_kind: str  # e.g. "move", "down", "up", "dwell"


@dataclass(frozen=True)
class SymbolBurst:
    """Channel-tagged symbol arrival or edit/revision burst."""
    ts_ms: float
    channel: str
    symbol_count: int = 1
    is_edit: bool = False
    edit_span: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel", normalize_channel(self.channel))
        if self.symbol_count < 0:
            raise ValueError("symbol_count must be >= 0")


@dataclass(frozen=True)
class SessionStream:
    """Unlabeled session stream of normalized 2D ticks and symbol bursts."""
    session_id: str
    ticks: tuple[TrajectoryTick, ...]
    bursts: tuple[SymbolBurst, ...]


FEATURE_NAMES: tuple[str, ...] = (
    # --- 2D Trajectory hardware-rate-agnostic geometric & ballistic invariants ---
    "path_tortuosity_log",
    "signed_curvature_mean",
    "signed_curvature_std",
    "curvature_skew",
    "peak_velocity_phase",
    "velocity_cv",
    "normalized_jerk_log",
    "xy_anisotropy_log",
    "terminal_overshoot_rate",
    "pre_act_dwell_ratio",
    # --- Channel-invariant burst rhythm & edit/revision kinematics ---
    "log_ibi_cv",
    "ibi_lag1_autocorr",
    "ibi_skewness",
    "long_pause_ratio",
    "edit_burst_rate",
    "pre_edit_hesitation_log",
    "post_edit_recovery_log",
    "edit_cluster_len_log",
    "traj_burst_coupling_log",
)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float], mu: float | None = None) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs) if mu is None else mu
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(max(var, 0.0))


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _skewness(xs: Sequence[float]) -> float:
    if len(xs) < 3:
        return 0.0
    m = _mean(xs)
    sd = _std(xs, m)
    if sd < 1e-9:
        return 0.0
    m3 = sum(((x - m) / sd) ** 3 for x in xs) / len(xs)
    return max(-5.0, min(5.0, m3))


def _lag1_autocorr(xs: Sequence[float]) -> float:
    if len(xs) < 3:
        return 0.0
    m = _mean(xs)
    denom = sum((x - m) ** 2 for x in xs)
    if denom < 1e-12:
        return 0.0
    num = sum((xs[i] - m) * (xs[i + 1] - m) for i in range(len(xs) - 1))
    return max(-1.0, min(1.0, num / denom))


def _resample_arc_length(
    ticks: Sequence[TrajectoryTick], num_points: int = 33
) -> list[tuple[float, float, float]]:
    """Resamples 2D trajectory ticks at uniform spatial arc-length intervals.

    Eliminates hardware polling-rate differences (e.g. 120 Hz mouse vs 15 Hz low-rate device)
    while preserving exact spatial geometry and normalized time progression.
    """
    if len(ticks) < 2:
        if not ticks:
            return [(0.5, 0.5, 0.0)] * num_points
        t0 = ticks[0]
        return [(t0.x_norm, t0.y_norm, 0.0)] * num_points

    cum_s = [0.0]
    for i in range(1, len(ticks)):
        dx = ticks[i].x_norm - ticks[i - 1].x_norm
        dy = ticks[i].y_norm - ticks[i - 1].y_norm
        cum_s.append(cum_s[-1] + math.hypot(dx, dy))

    total_s = cum_s[-1]
    t_start = ticks[0].ts_ms
    t_span = max(ticks[-1].ts_ms - t_start, 1.0)

    if total_s < 1e-9:
        t0 = ticks[0]
        return [
            (t0.x_norm, t0.y_norm, i / max(1, num_points - 1))
            for i in range(num_points)
        ]

    out: list[tuple[float, float, float]] = []
    seg_idx = 0
    for k in range(num_points):
        target_s = (k / (num_points - 1)) * total_s
        while seg_idx + 1 < len(cum_s) - 1 and cum_s[seg_idx + 1] < target_s:
            seg_idx += 1
        s0, s1 = cum_s[seg_idx], cum_s[seg_idx + 1]
        alpha = 0.0 if (s1 - s0) < 1e-12 else (target_s - s0) / (s1 - s0)
        alpha = max(0.0, min(1.0, alpha))
        a, b = ticks[seg_idx], ticks[seg_idx + 1]
        x = a.x_norm + alpha * (b.x_norm - a.x_norm)
        y = a.y_norm + alpha * (b.y_norm - a.y_norm)
        tn = ((a.ts_ms + alpha * (b.ts_ms - a.ts_ms)) - t_start) / t_span
        out.append((x, y, tn))
    return out


def _split_trajectory_strokes(
    ticks: Sequence[TrajectoryTick],
) -> list[list[TrajectoryTick]]:
    """Splits a tick sequence into individual ballistic pointer strokes.

    Stroke boundaries occur on non-'move' trigger actions or relative time gaps
    exceeding 3.5x the stream's own median tick interval (zero absolute Hz assumption).
    """
    if not ticks:
        return []
    dts = [
        ticks[i].ts_ms - ticks[i - 1].ts_ms
        for i in range(1, len(ticks))
        if ticks[i].ts_ms > ticks[i - 1].ts_ms
    ]
    med_dt = _median(dts) if dts else 16.0
    gap_thresh = max(med_dt * 3.5, 180.0)

    strokes: list[list[TrajectoryTick]] = []
    cur: list[TrajectoryTick] = [ticks[0]]
    for i in range(1, len(ticks)):
        dt = ticks[i].ts_ms - ticks[i - 1].ts_ms
        prev_act = ticks[i - 1].act_kind.lower()
        if dt > gap_thresh or prev_act in ("up", "click", "commit"):
            if len(cur) >= 3:
                strokes.append(cur)
            cur = [ticks[i]]
        else:
            cur.append(ticks[i])
    if len(cur) >= 3:
        strokes.append(cur)
    if not strokes and len(ticks) >= 2:
        strokes.append(list(ticks))
    return strokes


def _extract_trajectory_features(ticks: Sequence[TrajectoryTick]) -> list[float]:
    strokes = _split_trajectory_strokes(ticks)
    if not strokes:
        return [0.0] * 10

    tort_vals: list[float] = []
    curv_all: list[float] = []
    peak_phases: list[float] = []
    vel_cvs: list[float] = []
    jerk_logs: list[float] = []
    aniso_logs: list[float] = []
    overshoot_flags: list[float] = []
    dwell_ratios: list[float] = []

    for stroke in strokes:
        raw_pts = _resample_arc_length(stroke, num_points=25)
        # 3-tap spatial smoothing after uniform arc-length resampling eliminates
        # polling-rate quantization artifacts (120 Hz vs 15 Hz invariance).
        pts: list[tuple[float, float, float]] = [raw_pts[0]]
        for i in range(1, len(raw_pts) - 1):
            sx = 0.25 * raw_pts[i - 1][0] + 0.50 * raw_pts[i][0] + 0.25 * raw_pts[i + 1][0]
            sy = 0.25 * raw_pts[i - 1][1] + 0.50 * raw_pts[i][1] + 0.25 * raw_pts[i + 1][1]
            pts.append((sx, sy, raw_pts[i][2]))
        pts.append(raw_pts[-1])

        total_s = 0.0
        dxs: list[float] = []
        dys: list[float] = []
        norm_speeds: list[float] = []
        speed_times: list[float] = []

        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            ds = math.hypot(dx, dy)
            dt_n = max(pts[i][2] - pts[i - 1][2], 1e-3)
            total_s += ds
            dxs.append(dx)
            dys.append(dy)
            norm_speeds.append(ds / dt_n)
            speed_times.append(0.5 * (pts[i][2] + pts[i - 1][2]))

        chord = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
        tort_vals.append(math.log1p(max(0.0, total_s / max(chord, 1e-4) - 1.0)))

        # Macro-scale signed turning angles (stride=3 along arc-length) for Hz-invariant handedness bow
        macro_headings: list[float] = []
        stride = 3
        for i in range(0, len(pts) - stride, stride):
            mdx = pts[i + stride][0] - pts[i][0]
            mdy = pts[i + stride][1] - pts[i][1]
            if math.hypot(mdx, mdy) > 1e-6:
                macro_headings.append(math.atan2(mdy, mdx))
        stroke_curv: list[float] = []
        for i in range(1, len(macro_headings)):
            dtheta = macro_headings[i] - macro_headings[i - 1]
            while dtheta > math.pi:
                dtheta -= 2.0 * math.pi
            while dtheta < -math.pi:
                dtheta += 2.0 * math.pi
            stroke_curv.append(dtheta)
        curv_all.extend(stroke_curv)

        # Normalized speed profile (completely independent of absolute velocity / Hz)
        if norm_speeds:
            # Smooth speed profile across 3 bins before finding peak & derivative
            sm_v = [norm_speeds[0]]
            for i in range(1, len(norm_speeds) - 1):
                sm_v.append(0.25 * norm_speeds[i - 1] + 0.50 * norm_speeds[i] + 0.25 * norm_speeds[i + 1])
            sm_v.append(norm_speeds[-1])

            mean_v = max(_mean(sm_v), 1e-6)
            rel_v = [v / mean_v for v in sm_v]
            # Weighted centroid around top speeds gives smooth Hz-invariant peak phase
            v_sq = [v * v for v in rel_v]
            v_sq_sum = max(sum(v_sq), 1e-9)
            peak_phases.append(sum(t * w for t, w in zip(speed_times, v_sq)) / v_sq_sum)
            vel_cvs.append(_std(rel_v))
            acc = [rel_v[i] - rel_v[i - 1] for i in range(1, len(rel_v))]
            jerk_logs.append(math.log1p(_std(acc)))

        # X/Y displacement energy anisotropy
        dx_rms = math.sqrt(_mean([x * x for x in dxs]))
        dy_rms = math.sqrt(_mean([y * y for y in dys]))
        aniso_logs.append(math.log((dx_rms + 1e-4) / (dy_rms + 1e-4)))

        # Terminal overshoot / micro-correction in final 28% of stroke
        if len(pts) >= 5 and chord > 1e-4:
            ux = (pts[-1][0] - pts[0][0]) / chord
            uy = (pts[-1][1] - pts[0][1]) / chord
            tail_start = max(1, int(len(pts) * 0.72))
            rev = 0
            tot_tail = 0
            for i in range(tail_start, len(pts)):
                proj = (pts[i][0] - pts[i - 1][0]) * ux + (pts[i][1] - pts[i - 1][1]) * uy
                tot_tail += 1
                if proj < -1e-5:
                    rev += 1
            overshoot_flags.append(rev / max(1, tot_tail))

        # Pre-action dwell ratio measured in normalized time (independent of tick rate)
        stroke_dur = max(stroke[-1].ts_ms - stroke[0].ts_ms, 1.0)
        first_dwell_ts = None
        for tk in stroke:
            if tk.act_kind.lower() in ("dwell", "down", "click"):
                first_dwell_ts = tk.ts_ms
                break
        if first_dwell_ts is not None:
            dwell_ratios.append(min(1.0, max(0.0, (stroke[-1].ts_ms - first_dwell_ts) / stroke_dur)))
        else:
            dwell_ratios.append(0.0)

    return [
        _mean(tort_vals),
        _mean(curv_all),
        _std(curv_all),
        _skewness(curv_all),
        _mean(peak_phases),
        _mean(vel_cvs),
        _mean(jerk_logs),
        _mean(aniso_logs),
        _mean(overshoot_flags),
        _mean(dwell_ratios),
    ]


def _extract_burst_features(
    ticks: Sequence[TrajectoryTick], bursts: Sequence[SymbolBurst]
) -> list[float]:
    if len(bursts) < 3:
        return [0.0] * 9

    ordered = sorted(bursts, key=lambda b: b.ts_ms)
    ibis: list[float] = []
    normal_ibis: list[float] = []
    pre_edit_ibis: list[float] = []
    post_edit_ibis: list[float] = []
    edit_runs: list[int] = []

    run_len = 0
    for i, b in enumerate(ordered):
        if b.is_edit:
            run_len += 1
        elif run_len > 0:
            edit_runs.append(run_len)
            run_len = 0

        if i == 0:
            continue
        dt = max(ordered[i].ts_ms - ordered[i - 1].ts_ms, 1.0)
        # Normalize per-burst interval without assuming channel symbol_count=1
        ibis.append(dt)
        prev_edit = ordered[i - 1].is_edit
        if b.is_edit and not prev_edit:
            pre_edit_ibis.append(dt)
        elif not b.is_edit and prev_edit:
            post_edit_ibis.append(dt)
        elif not b.is_edit and not prev_edit:
            normal_ibis.append(dt)

    if run_len > 0:
        edit_runs.append(run_len)

    med_normal = _median(normal_ibis) if normal_ibis else _median(ibis)
    med_normal = max(med_normal, 1.0)

    # Scale-invariant dimensionless log-ratios: immune to fatigue uniform slowdown!
    rel_log_ibis = [math.log(dt / med_normal) for dt in ibis]
    log_ibi_cv = _std(rel_log_ibis)
    ibi_lag1 = _lag1_autocorr(rel_log_ibis)
    ibi_skew = _skewness(rel_log_ibis)
    long_pause_ratio = sum(1 for dt in ibis if dt > 2.2 * med_normal) / len(ibis)

    edit_count = sum(1 for b in ordered if b.is_edit)
    edit_rate = edit_count / len(ordered)

    pre_edit_ratio = (
        math.log((_median(pre_edit_ibis) / med_normal) + 1e-3)
        if pre_edit_ibis
        else 0.0
    )
    post_edit_ratio = (
        math.log((_median(post_edit_ibis) / med_normal) + 1e-3)
        if post_edit_ibis
        else 0.0
    )
    edit_cluster_log = math.log1p(_mean([float(r) for r in edit_runs])) if edit_runs else 0.0

    # Trajectory-to-burst handoff coupling (dimensionless ratio to median IBI)
    coupling_ratios: list[float] = []
    if ticks:
        action_ts = sorted(
            t.ts_ms for t in ticks if t.act_kind.lower() in ("down", "up", "click", "dwell")
        )
        b_idx = 0
        for ats in action_ts:
            while b_idx < len(ordered) and ordered[b_idx].ts_ms < ats:
                b_idx += 1
            if b_idx < len(ordered):
                gap = ordered[b_idx].ts_ms - ats
                if 0.0 < gap <= 12.0 * med_normal:
                    coupling_ratios.append(math.log((gap / med_normal) + 0.05))
    coupling_log = _mean(coupling_ratios) if coupling_ratios else 0.0

    return [
        log_ibi_cv,
        ibi_lag1,
        ibi_skew,
        long_pause_ratio,
        edit_rate,
        pre_edit_ratio,
        post_edit_ratio,
        edit_cluster_log,
        coupling_log,
    ]


def extract_features(stream: SessionStream) -> list[float]:
    """Extracts the 19-dimensional hardware-rate & channel-invariant feature vector."""
    traj = _extract_trajectory_features(stream.ticks)
    burst = _extract_burst_features(stream.ticks, stream.bursts)
    vec = traj + burst
    assert len(vec) == len(FEATURE_NAMES)
    return vec


def split_sequential_windows(
    stream: SessionStream, num_windows: int = 3
) -> list[SessionStream]:
    """Splits a session chronologically into overlapping sequential windows for CUSUM scoring."""
    if num_windows <= 1 or (len(stream.ticks) < 12 and len(stream.bursts) < 12):
        return [stream]
    all_ts = [t.ts_ms for t in stream.ticks] + [b.ts_ms for b in stream.bursts]
    if not all_ts:
        return [stream]
    t_min, t_max = min(all_ts), max(all_ts)
    span = max(t_max - t_min, 1.0)
    win_len = span * 0.55
    step = (span - win_len) / max(1, num_windows - 1)

    windows: list[SessionStream] = []
    for k in range(num_windows):
        lo = t_min + k * step
        hi = lo + win_len + 1e-6
        wt = tuple(t for t in stream.ticks if lo <= t.ts_ms <= hi)
        wb = tuple(b for b in stream.bursts if lo <= b.ts_ms <= hi)
        if len(wt) >= 4 and len(wb) >= 4:
            windows.append(SessionStream(f"{stream.session_id}#w{k}", wt, wb))
    return windows if windows else [stream]


def _invert_matrix(mat: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan matrix inversion with partial pivoting (pure Python stdlib)."""
    n = len(mat)
    aug = [
        [float(mat[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError("Matrix is singular after regularization")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        piv = aug[col][col]
        aug[col] = [v / piv for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor != 0.0:
                aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(2 * n)]
    return [row[n:] for row in aug]


def _mahalanobis(vec: Sequence[float], mean: Sequence[float], inv_cov: Sequence[Sequence[float]]) -> float:
    d = len(vec)
    diff = [vec[i] - mean[i] for i in range(d)]
    total = 0.0
    for i in range(d):
        row = inv_cov[i]
        s = 0.0
        for j in range(d):
            s += row[j] * diff[j]
        total += diff[i] * s
    return math.sqrt(max(total, 0.0) / max(1, d))


@dataclass(frozen=True)
class ScoreResult:
    session_id: str
    global_distance: float
    sequential_cusum: float
    combined_score: float
    threshold: float
    accepted: bool
    top_deviations: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ProfileModel:
    """Frozen enrolled and pre-test-calibrated profile."""
    profile_id: str
    mean: tuple[float, ...]
    stds: tuple[float, ...]
    inv_cov: tuple[tuple[float, ...], ...]
    cusum_ref: float
    threshold: float
    enroll_count: int
    calibration_genuine_count: int
    calibration_impostor_count: int
    calibration_locked: bool = True

    def score(self, stream: SessionStream) -> ScoreResult:
        vec = extract_features(stream)
        global_dist = _mahalanobis(vec, self.mean, self.inv_cov)

        # Sequential windowed CUSUM + worst-window detector catches mid-session hijacks
        windows = split_sequential_windows(stream, num_windows=4)
        cusum = 0.0
        cusum_max = 0.0
        worst_win = global_dist
        for w in windows:
            wvec = extract_features(w)
            wd = _mahalanobis(wvec, self.mean, self.inv_cov)
            if wd > worst_win:
                worst_win = wd
            cusum = max(0.0, cusum + (wd - self.cusum_ref))
            if cusum > cusum_max:
                cusum_max = cusum

        combined = 0.60 * global_dist + 0.25 * worst_win + 0.15 * (cusum_max / max(1.0, len(windows) * 0.5))

        z_scores = sorted(
            (
                (FEATURE_NAMES[i], abs(vec[i] - self.mean[i]) / max(self.stds[i], 1e-4))
                for i in range(len(vec))
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return ScoreResult(
            session_id=stream.session_id,
            global_distance=global_dist,
            sequential_cusum=cusum_max,
            combined_score=combined,
            threshold=self.threshold,
            accepted=(combined <= self.threshold),
            top_deviations=tuple(z_scores[:3]),
        )


def fit_uncalibrated_profile(
    profile_id: str,
    enroll_streams: Sequence[SessionStream],
    shrinkage: float = 0.45,
    min_var: float = 2e-5,
) -> ProfileModel:
    """Fits mean + shrinkage-regularized inverse covariance strictly on enrollment data."""
    if len(enroll_streams) < 4:
        raise ValueError("Need at least 4 enrollment sessions")
    feats = [extract_features(s) for s in enroll_streams]
    d = len(feats[0])
    n = len(feats)

    mean = [sum(f[j] for f in feats) / n for j in range(d)]
    # Sample covariance
    cov = [[0.0] * d for _ in range(d)]
    for f in feats:
        diff = [f[j] - mean[j] for j in range(d)]
        for i in range(d):
            di = diff[i]
            for j in range(d):
                cov[i][j] += di * diff[j]
    denom = max(1, n - 1)
    for i in range(d):
        for j in range(d):
            cov[i][j] /= denom

    # Ledoit-Wolf style diagonal shrinkage + variance floor (prevents overfitting & handles fatigue)
    reg_cov = [[0.0] * d for _ in range(d)]
    stds = []
    for i in range(d):
        var_i = max(cov[i][i], min_var)
        stds.append(math.sqrt(var_i))
        for j in range(d):
            if i == j:
                reg_cov[i][j] = var_i * (1.0 + 0.15)
            else:
                reg_cov[i][j] = (1.0 - shrinkage) * cov[i][j]

    inv_cov = _invert_matrix(reg_cov)

    # Set CUSUM reference drift allowance to the 85th percentile of enrollment window distances
    win_dists: list[float] = []
    for s in enroll_streams:
        for w in split_sequential_windows(s, num_windows=4):
            win_dists.append(_mahalanobis(extract_features(w), mean, inv_cov))
    win_dists.sort()
    ref_idx = min(len(win_dists) - 1, int(0.85 * len(win_dists)))
    cusum_ref = win_dists[ref_idx] if win_dists else 1.25

    return ProfileModel(
        profile_id=profile_id,
        mean=tuple(mean),
        stds=tuple(stds),
        inv_cov=tuple(tuple(r) for r in inv_cov),
        cusum_ref=cusum_ref,
        threshold=float("inf"),
        enroll_count=n,
        calibration_genuine_count=0,
        calibration_impostor_count=0,
        calibration_locked=False,
    )


def calibrate_profile(
    uncalibrated: ProfileModel,
    calib_genuine: Sequence[SessionStream],
    calib_impostor: Sequence[SessionStream],
    far_weight: float = 1.6,
) -> ProfileModel:
    """Calibrates and freezes the decision threshold strictly on pre-test calibration streams.

    Never sees holdout evaluation streams; selects the threshold minimizing
    weighted calibration error (far_weight * FAR + FRR) with maximum margin tie-breaking.
    """
    if uncalibrated.calibration_locked:
        raise ValueError("Profile threshold is already calibrated and locked")
    if not calib_genuine or not calib_impostor:
        raise ValueError("Both calibration genuine and impostor sets are required")

    g_scores = sorted(uncalibrated.score(s).combined_score for s in calib_genuine)
    i_scores = sorted(uncalibrated.score(s).combined_score for s in calib_impostor)

    all_pts = sorted(set(g_scores + i_scores))
    candidates = [0.5 * (all_pts[k] + all_pts[k + 1]) for k in range(len(all_pts) - 1)]
    if not candidates:
        candidates = [all_pts[0]]

    best_tau = candidates[0]
    best_obj = float("inf")
    best_margin = -float("inf")

    for tau in candidates:
        frr = sum(1 for s in g_scores if s > tau) / len(g_scores)
        far = sum(1 for s in i_scores if s <= tau) / len(i_scores)
        obj = far_weight * far + frr

        # Margin to closest genuine below tau and closest impostor above tau
        below = [s for s in g_scores if s <= tau]
        above = [s for s in i_scores if s > tau]
        margin = (
            min(tau - max(below), min(above) - tau)
            if (below and above)
            else -abs(tau - _median(g_scores))
        )
        if obj < best_obj - 1e-12 or (abs(obj - best_obj) <= 1e-12 and margin > best_margin):
            best_obj = obj
            best_margin = margin
            best_tau = tau

    return ProfileModel(
        profile_id=uncalibrated.profile_id,
        mean=uncalibrated.mean,
        stds=uncalibrated.stds,
        inv_cov=uncalibrated.inv_cov,
        cusum_ref=uncalibrated.cusum_ref,
        threshold=best_tau,
        enroll_count=uncalibrated.enroll_count,
        calibration_genuine_count=len(calib_genuine),
        calibration_impostor_count=len(calib_impostor),
        calibration_locked=True,
    )


@dataclass(frozen=True)
class EvalReport:
    profile_id: str
    threshold: float
    normal_frr_num: int
    normal_frr_den: int
    fatigued_frr_num: int
    fatigued_frr_den: int
    stranger_far_num: int
    stranger_far_den: int
    mimic_far_num: int
    mimic_far_den: int

    @property
    def total_frr(self) -> float:
        den = self.normal_frr_den + self.fatigued_frr_den
        return (self.normal_frr_num + self.fatigued_frr_num) / max(1, den)

    @property
    def total_far(self) -> float:
        den = self.stranger_far_den + self.mimic_far_den
        return (self.stranger_far_num + self.mimic_far_num) / max(1, den)


def evaluate_holdout(
    model: ProfileModel,
    holdout_normal: Sequence[SessionStream],
    holdout_fatigued: Sequence[SessionStream],
    holdout_stranger: Sequence[SessionStream],
    holdout_mimic: Sequence[SessionStream],
) -> EvalReport:
    if not model.calibration_locked:
        raise ValueError("Cannot evaluate holdout on an uncalibrated model")
    n_rej = sum(1 for s in holdout_normal if not model.score(s).accepted)
    f_rej = sum(1 for s in holdout_fatigued if not model.score(s).accepted)
    s_acc = sum(1 for s in holdout_stranger if model.score(s).accepted)
    m_acc = sum(1 for s in holdout_mimic if model.score(s).accepted)
    return EvalReport(
        profile_id=model.profile_id,
        threshold=model.threshold,
        normal_frr_num=n_rej,
        normal_frr_den=len(holdout_normal),
        fatigued_frr_num=f_rej,
        fatigued_frr_den=len(holdout_fatigued),
        stranger_far_num=s_acc,
        stranger_far_den=len(holdout_stranger),
        mimic_far_num=m_acc,
        mimic_far_den=len(holdout_mimic),
    )


def evaluate_swap_matrix(
    models: Sequence[ProfileModel],
    genuine_holdout_by_profile: dict[str, Sequence[SessionStream]],
) -> dict[str, dict[str, tuple[int, int]]]:
    """Full N x N cross-profile swap matrix: cell[model_A][sessions_from_B] = (accepted, total)."""
    matrix: dict[str, dict[str, tuple[int, int]]] = {}
    for m in models:
        row: dict[str, tuple[int, int]] = {}
        for other in models:
            streams = genuine_holdout_by_profile[other.profile_id]
            acc = sum(1 for s in streams if m.score(s).accepted)
            row[other.profile_id] = (acc, len(streams))
        matrix[m.profile_id] = row
    return matrix


if __name__ == "__main__":
    from synth_traces import run_full_benchmark

    raise SystemExit(run_full_benchmark())
