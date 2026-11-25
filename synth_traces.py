"""synth_traces.py — Deterministic zero-dependency synthetic benchmark generator
and end-to-end FAR/FRR + NxN swap-matrix runner for kinematics_auth.py (t=23).

Covers 3 distinct interaction styles (>= 2 required):
  1. fast_continuous_direct : High-rate (~100-120 Hz) continuous trajectories + DIRECT_KEY strokes
  2. slow_assist_batch      : Low-rate (~14-18 Hz) slow trajectories + BATCH_INSERT text bursts
  3. hybrid_virtual_grid    : Mid-rate (~40-50 Hz) dwell-heavy trajectories + VIRTUAL_GRID bursts

Strict 3-way split with disjoint RNG seeds (zero test-set leakage):
  - enroll      : 14 enrolled_normal sessions
  - calibration : 12 genuine (8 normal + 4 fatigued) & 16 impostor (8 stranger + 8 habit_mimic)
  - holdout     : 18 genuine (12 normal + 6 fatigued) & 24 impostor (12 stranger + 12 habit_mimic)
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

from kinematics_auth import (
    EvalReport,
    ProfileModel,
    SessionStream,
    SymbolBurst,
    TrajectoryTick,
    calibrate_profile,
    evaluate_holdout,
    evaluate_swap_matrix,
    fit_uncalibrated_profile,
)


@dataclass(frozen=True)
class StyleSpec:
    profile_id: str
    channel: str
    poll_hz: float
    stroke_dur_ms: float
    base_ibi_ms: float
    batch_size_range: tuple[int, int]
    # Personal neuromuscular & burst-rhythm invariants
    bow_amp: float            # Signed arc bow (+left / -right handedness bias)
    peak_phase: float         # Normalized time of peak velocity in (0.25, 0.75)
    overshoot_amp: float      # Terminal micro-correction amplitude
    pre_act_dwell_frac: float # Dwell time fraction before click/commit
    xy_ratio: float           # Horizontal vs vertical motion bias
    ibi_sigma: float          # Log-IBI relative dispersion
    ibi_ar1: float            # Lag-1 autocorrelation of log-IBI rhythm
    long_pause_prob: float    # Cognitive pause probability
    edit_prob: float          # Edit/revision burst probability
    pre_edit_mult: float      # Hesitation latency multiplier before edit
    post_edit_mult: float     # Recovery latency multiplier after edit
    edit_run_prob: float      # Geometric continuation prob for multi-edit bursts
    handoff_mult: float       # Pointer-to-symbol handoff ratio relative to base_ibi


STYLE_SPECS: tuple[StyleSpec, ...] = (
    StyleSpec(
        profile_id="fast_continuous_direct",
        channel="DIRECT_KEY",
        poll_hz=110.0,
        stroke_dur_ms=320.0,
        base_ibi_ms=115.0,
        batch_size_range=(1, 1),
        bow_amp=0.075,
        peak_phase=0.33,
        overshoot_amp=0.042,
        pre_act_dwell_frac=0.06,
        xy_ratio=1.65,
        ibi_sigma=0.22,
        ibi_ar1=0.58,
        long_pause_prob=0.05,
        edit_prob=0.14,
        pre_edit_mult=1.55,
        post_edit_mult=0.78,
        edit_run_prob=0.55,
        handoff_mult=1.40,
    ),
    StyleSpec(
        profile_id="slow_assist_batch",
        channel="BATCH_INSERT",
        poll_hz=16.0,
        stroke_dur_ms=1450.0,
        base_ibi_ms=980.0,
        batch_size_range=(4, 14),
        bow_amp=-0.068,
        peak_phase=0.63,
        overshoot_amp=0.008,
        pre_act_dwell_frac=0.26,
        xy_ratio=0.72,
        ibi_sigma=0.34,
        ibi_ar1=-0.28,
        long_pause_prob=0.18,
        edit_prob=0.09,
        pre_edit_mult=2.45,
        post_edit_mult=1.95,
        edit_run_prob=0.18,
        handoff_mult=3.10,
    ),
    StyleSpec(
        profile_id="hybrid_virtual_grid",
        channel="VIRTUAL_GRID",
        poll_hz=45.0,
        stroke_dur_ms=720.0,
        base_ibi_ms=340.0,
        batch_size_range=(1, 3),
        bow_amp=0.012,
        peak_phase=0.49,
        overshoot_amp=0.022,
        pre_act_dwell_frac=0.17,
        xy_ratio=1.10,
        ibi_sigma=0.48,
        ibi_ar1=0.18,
        long_pause_prob=0.26,
        edit_prob=0.24,
        pre_edit_mult=0.88,
        post_edit_mult=1.42,
        edit_run_prob=0.72,
        handoff_mult=0.65,
    ),
)


def _warped_progress(u: float, peak_phase: float) -> float:
    """Smooth monotone s-curve whose peak velocity occurs near `peak_phase`."""
    # Power warp maps u in [0,1] so peak derivative shifts left (<0.5) or right (>0.5)
    gamma = math.log(0.5) / math.log(max(0.15, min(0.85, peak_phase)))
    w = u ** gamma
    return w * w * (3.0 - 2.0 * w)


def _generate_session(
    spec: StyleSpec,
    session_id: str,
    rng: random.Random,
    *,
    mode: str,
    poll_hz_override: float | None = None,
    channel_override: str | None = None,
) -> SessionStream:
    """Generates one synthetic session stream under one of four regimes:
    'enrolled_normal', 'enrolled_fatigued', 'stranger', or 'habit_mimic'.
    """
    # Base parameters
    poll_hz = poll_hz_override if poll_hz_override is not None else spec.poll_hz
    channel = channel_override if channel_override is not None else spec.channel
    stroke_dur = spec.stroke_dur_ms
    base_ibi = spec.base_ibi_ms
    bow = spec.bow_amp
    peak_p = spec.peak_phase
    overshoot = spec.overshoot_amp
    dwell_f = spec.pre_act_dwell_frac
    xy_ratio = spec.xy_ratio
    ibi_sigma = spec.ibi_sigma
    ibi_ar1 = spec.ibi_ar1
    pause_p = spec.long_pause_prob
    edit_p = spec.edit_prob
    pre_edit = spec.pre_edit_mult
    post_edit = spec.post_edit_mult
    edit_run = spec.edit_run_prob
    handoff = spec.handoff_mult
    noise_scale = 1.0

    if mode == "enrolled_fatigued":
        # Fatigued genuine operator: 28%-42% slower overall, mild jitter increase,
        # but preserves core neuromuscular bow, peak phase, edit latency ratios & AR(1) rhythm.
        slowdown = rng.uniform(1.28, 1.44)
        stroke_dur *= slowdown
        base_ibi *= slowdown
        noise_scale = 1.18
        pause_p = min(0.45, pause_p * 1.12)
    elif mode == "habit_mimic":
        # Adversarial parameter-matched mimic: matches exact poll_hz, channel, stroke_dur & base_ibi,
        # but fails to replicate subtle handedness bow, peak velocity phase, AR(1) coupling & edit ratios.
        bow = -bow * rng.uniform(0.75, 1.25) + rng.gauss(0.0, 0.015)
        peak_p = max(0.22, min(0.78, 1.0 - peak_p + rng.gauss(0.0, 0.04)))
        overshoot = max(0.0, 0.045 - overshoot + rng.gauss(0.0, 0.005))
        dwell_f = max(0.03, min(0.40, dwell_f + rng.choice((-0.11, 0.11))))
        ibi_ar1 = max(-0.65, min(0.75, -ibi_ar1 + rng.gauss(0.0, 0.08)))
        pre_edit = max(0.55, pre_edit * rng.choice((0.62, 1.55)))
        post_edit = max(0.55, post_edit * rng.choice((0.60, 1.60)))
        edit_run = max(0.08, min(0.85, 0.85 - edit_run))
        handoff = max(0.35, handoff * rng.choice((0.55, 1.75)))
    elif mode == "stranger":
        # Completely unrelated operator using arbitrary random kinematics & rhythm
        stroke_dur *= rng.uniform(0.55, 1.85)
        base_ibi *= rng.uniform(0.55, 1.85)
        bow = rng.uniform(-0.11, 0.11)
        if abs(bow - spec.bow_amp) < 0.035:
            bow = -spec.bow_amp
        peak_p = rng.uniform(0.22, 0.78)
        if abs(peak_p - spec.peak_phase) < 0.12:
            peak_p = 1.0 - spec.peak_phase
        overshoot = rng.uniform(0.0, 0.06)
        dwell_f = rng.uniform(0.03, 0.36)
        xy_ratio = rng.uniform(0.55, 2.10)
        ibi_sigma = rng.uniform(0.14, 0.62)
        ibi_ar1 = rng.uniform(-0.55, 0.75)
        pause_p = rng.uniform(0.02, 0.35)
        edit_p = rng.uniform(0.03, 0.32)
        pre_edit = rng.uniform(0.70, 2.80)
        post_edit = rng.uniform(0.65, 2.40)
        edit_run = rng.uniform(0.10, 0.80)
        handoff = rng.uniform(0.45, 3.60)
    elif mode != "enrolled_normal":
        raise ValueError(f"Unknown mode: {mode}")

    ticks: list[TrajectoryTick] = []
    bursts: list[SymbolBurst] = []
    t_cursor = rng.uniform(0.0, 50.0)
    num_blocks = 6

    for blk in range(num_blocks):
        # 1. Generate 2 ballistic strokes per block
        for _ in range(2):
            dur = max(120.0, stroke_dur * rng.uniform(0.85, 1.18))
            dt_step = 1000.0 / max(8.0, poll_hz * rng.uniform(0.92, 1.08))
            n_steps = max(7, int(dur / dt_step))

            dx_span = rng.uniform(0.18, 0.42) * math.sqrt(xy_ratio) * rng.choice((-1.0, 1.0))
            dy_span = rng.uniform(0.18, 0.42) / math.sqrt(xy_ratio) * rng.choice((-1.0, 1.0))
            x0 = rng.uniform(0.25, 0.75)
            y0 = rng.uniform(0.25, 0.75)
            x1 = max(0.05, min(0.95, x0 + dx_span))
            y1 = max(0.05, min(0.95, y0 + dy_span))
            dx, dy = x1 - x0, y1 - y0
            chord = max(math.hypot(dx, dy), 1e-4)
            # Perpendicular unit normal for signed handedness curvature
            nx, ny = -dy / chord, dx / chord

            dwell_steps = max(1, int(round(n_steps * dwell_f)))
            move_steps = max(5, n_steps - dwell_steps)
            block_bow = bow * rng.uniform(0.82, 1.18)
            block_peak = max(0.18, min(0.82, peak_p + rng.gauss(0.0, 0.025 * noise_scale)))

            for k in range(move_steps):
                u = k / max(1, move_steps - 1)
                p = _warped_progress(u, block_peak)
                # Smooth terminal micro-overshoot wave in final 25%
                over = 0.0
                if overshoot > 0.005 and u > 0.72:
                    phase_o = (u - 0.72) / 0.28
                    over = overshoot * math.sin(phase_o * math.pi) * (1.0 if phase_o < 0.55 else -0.6)
                lat = block_bow * chord * math.sin(math.pi * p)
                jx = rng.gauss(0.0, 0.0018 * noise_scale)
                jy = rng.gauss(0.0, 0.0018 * noise_scale)
                px = max(0.0, min(1.0, x0 + (p + over) * dx + lat * nx + jx))
                py = max(0.0, min(1.0, y0 + (p + over) * dy + lat * ny + jy))
                ticks.append(TrajectoryTick(t_cursor, px, py, "move"))
                t_cursor += dt_step * rng.uniform(0.94, 1.06)

            for d_i in range(dwell_steps):
                act = "down" if d_i == dwell_steps - 1 else "dwell"
                px = max(0.0, min(1.0, x1 + rng.gauss(0.0, 0.0008 * noise_scale)))
                py = max(0.0, min(1.0, y1 + rng.gauss(0.0, 0.0008 * noise_scale)))
                ticks.append(TrajectoryTick(t_cursor, px, py, act))
                t_cursor += dt_step * rng.uniform(0.94, 1.06)

            ticks.append(TrajectoryTick(t_cursor + 8.0, x1, y1, "up"))
            t_cursor += max(dur * 0.45, 220.0)

        # 2. Handoff from pointer action to symbol burst sequence
        t_cursor += max(20.0, base_ibi * handoff * rng.uniform(0.82, 1.18))

        # 3. Generate channel-tagged symbol bursts + edit/revision clusters
        n_bursts = 14 if blk < num_blocks - 1 else 16
        z_prev = rng.gauss(0.0, ibi_sigma * noise_scale)
        in_edit_run = False
        prev_was_edit = False

        for _ in range(n_bursts):
            if in_edit_run:
                is_edit = rng.random() < edit_run
                in_edit_run = is_edit
            else:
                is_edit = rng.random() < edit_p
                in_edit_run = is_edit

            innov = rng.gauss(0.0, ibi_sigma * math.sqrt(max(0.05, 1.0 - ibi_ar1 * ibi_ar1)) * noise_scale)
            z_cur = ibi_ar1 * z_prev + innov
            z_prev = z_cur

            dt = base_ibi * math.exp(max(-1.4, min(1.8, z_cur)))
            if is_edit and not prev_was_edit:
                dt *= pre_edit * rng.uniform(0.88, 1.12)
            elif not is_edit and prev_was_edit:
                dt *= post_edit * rng.uniform(0.88, 1.12)
            elif rng.random() < pause_p:
                dt *= rng.uniform(2.35, 3.60)

            t_cursor += max(15.0, dt)
            lo_c, hi_c = spec.batch_size_range
            sym_cnt = rng.randint(lo_c, hi_c) if not is_edit else 0
            edit_span = rng.randint(1, max(1, hi_c // 2)) if is_edit else 0
            bursts.append(
                SymbolBurst(
                    ts_ms=t_cursor,
                    channel=channel,
                    symbol_count=sym_cnt,
                    is_edit=is_edit,
                    edit_span=edit_span,
                )
            )
            prev_was_edit = is_edit

        t_cursor += max(300.0, base_ibi * 1.5)

    return SessionStream(
        session_id=session_id,
        ticks=tuple(ticks),
        bursts=tuple(bursts),
    )


@dataclass(frozen=True)
class ProfileDataset:
    spec: StyleSpec
    enroll_normal: tuple[SessionStream, ...]
    calib_normal: tuple[SessionStream, ...]
    calib_fatigued: tuple[SessionStream, ...]
    calib_stranger: tuple[SessionStream, ...]
    calib_mimic: tuple[SessionStream, ...]
    holdout_normal: tuple[SessionStream, ...]
    holdout_fatigued: tuple[SessionStream, ...]
    holdout_stranger: tuple[SessionStream, ...]
    holdout_mimic: tuple[SessionStream, ...]


def build_profile_dataset(spec: StyleSpec, base_seed: int) -> ProfileDataset:
    """Constructs strictly disjoint enroll, calibration, and holdout splits."""
    def make_batch(tag: str, mode: str, count: int, seed_offset: int) -> tuple[SessionStream, ...]:
        rng = random.Random(base_seed + seed_offset)
        return tuple(
            _generate_session(spec, f"{spec.profile_id}:{tag}:{i}", rng, mode=mode)
            for i in range(count)
        )

    return ProfileDataset(
        spec=spec,
        enroll_normal=make_batch("enroll", "enrolled_normal", 16, 100),
        calib_normal=make_batch("cal_norm", "enrolled_normal", 10, 200),
        calib_fatigued=make_batch("cal_fat", "enrolled_fatigued", 6, 300),
        calib_stranger=make_batch("cal_str", "stranger", 10, 400),
        calib_mimic=make_batch("cal_mim", "habit_mimic", 10, 500),
        holdout_normal=make_batch("test_norm", "enrolled_normal", 12, 1100),
        holdout_fatigued=make_batch("test_fat", "enrolled_fatigued", 6, 1200),
        holdout_stranger=make_batch("test_str", "stranger", 12, 1300),
        holdout_mimic=make_batch("test_mim", "habit_mimic", 12, 1400),
    )


def run_full_benchmark(seed: int = 20261125) -> int:
    """Runs enrollment, strict pre-holdout calibration, holdout FAR/FRR evaluation,
    and full N x N cross-profile swap matrix verification.
    """
    datasets = [
        build_profile_dataset(spec, seed + idx * 10_000)
        for idx, spec in enumerate(STYLE_SPECS)
    ]

    models: list[ProfileModel] = []
    reports: list[EvalReport] = []
    genuine_holdout_map: dict[str, tuple[SessionStream, ...]] = {}

    for ds in datasets:
        uncal = fit_uncalibrated_profile(ds.spec.profile_id, ds.enroll_normal)
        calib_genuine = ds.calib_normal + ds.calib_fatigued
        calib_impostor = ds.calib_stranger + ds.calib_mimic
        model = calibrate_profile(uncal, calib_genuine, calib_impostor)
        models.append(model)

        rep = evaluate_holdout(
            model,
            holdout_normal=ds.holdout_normal,
            holdout_fatigued=ds.holdout_fatigued,
            holdout_stranger=ds.holdout_stranger,
            holdout_mimic=ds.holdout_mimic,
        )
        reports.append(rep)
        genuine_holdout_map[ds.spec.profile_id] = ds.holdout_normal + ds.holdout_fatigued

    swap = evaluate_swap_matrix(models, genuine_holdout_map)

    # Aggregate metrics
    tot_norm_rej = sum(r.normal_frr_num for r in reports)
    tot_norm_den = sum(r.normal_frr_den for r in reports)
    tot_fat_rej = sum(r.fatigued_frr_num for r in reports)
    tot_fat_den = sum(r.fatigued_frr_den for r in reports)
    tot_str_acc = sum(r.stranger_far_num for r in reports)
    tot_str_den = sum(r.stranger_far_den for r in reports)
    tot_mim_acc = sum(r.mimic_far_num for r in reports)
    tot_mim_den = sum(r.mimic_far_den for r in reports)

    tot_frr = (tot_norm_rej + tot_fat_rej) / max(1, tot_norm_den + tot_fat_den)
    tot_far = (tot_str_acc + tot_mim_acc) / max(1, tot_str_den + tot_mim_den)

    off_diag_unlocks = 0
    off_diag_total = 0
    diag_unlocks = 0
    diag_total = 0
    for pid_a, row in swap.items():
        for pid_b, (acc, tot) in row.items():
            if pid_a == pid_b:
                diag_unlocks += acc
                diag_total += tot
            else:
                off_diag_unlocks += acc
                off_diag_total += tot

    print("=== Multi-Channel 2D Kinematics & Burst-Rhythm Anomaly Verifier (t=23) ===")
    print(f"Styles evaluated: {len(models)} | Pre-holdout calibration locked: {all(m.calibration_locked for m in models)}")
    print()
    print("Profile                      | Fixed Tau | Normal FRR | Fatigued FRR | Stranger FAR | Mimic FAR  | Total FRR | Total FAR")
    print("-" * 118)
    for r in reports:
        print(
            f"{r.profile_id:<28} | {r.threshold:>9.4f} | "
            f"{r.normal_frr_num:>2}/{r.normal_frr_den:<2} ({100*r.normal_frr_num/r.normal_frr_den:>4.1f}%) | "
            f"{r.fatigued_frr_num:>2}/{r.fatigued_frr_den:<2} ({100*r.fatigued_frr_num/r.fatigued_frr_den:>5.1f}%) | "
            f"{r.stranger_far_num:>2}/{r.stranger_far_den:<2} ({100*r.stranger_far_num/r.stranger_far_den:>5.1f}%) | "
            f"{r.mimic_far_num:>2}/{r.mimic_far_den:<2} ({100*r.mimic_far_num/r.mimic_far_den:>4.1f}%) | "
            f"{100*r.total_frr:>8.2f}% | {100*r.total_far:>8.2f}%"
        )
    print("-" * 118)
    print(
        f"{'AGGREGATE':<28} | {'(locked)':>9} | "
        f"{tot_norm_rej:>2}/{tot_norm_den:<2} ({100*tot_norm_rej/tot_norm_den:>4.1f}%) | "
        f"{tot_fat_rej:>2}/{tot_fat_den:<2} ({100*tot_fat_rej/tot_fat_den:>5.1f}%) | "
        f"{tot_str_acc:>2}/{tot_str_den:<2} ({100*tot_str_acc/tot_str_den:>5.1f}%) | "
        f"{tot_mim_acc:>2}/{tot_mim_den:<2} ({100*tot_mim_acc/tot_mim_den:>4.1f}%) | "
        f"{100*tot_frr:>8.2f}% | {100*tot_far:>8.2f}%"
    )
    print()
    print("=== Full N x N Cross-Profile Swap Matrix (accepted / total genuine holdout sessions) ===")
    corner = "Model / Session"
    header = f"{corner:<26} | " + " | ".join(f"{m.profile_id:<22}" for m in models)
    print(header)
    print("-" * len(header))
    for m in models:
        cells = []
        for other in models:
            acc, tot = swap[m.profile_id][other.profile_id]
            cells.append(f"{acc:>2}/{tot:<2} ({100*acc/tot:>5.1f}%)        ")
        print(f"{m.profile_id:<26} | " + " | ".join(cells))
    print()
    print(
        f"Swap Matrix Check: diagonal={diag_unlocks}/{diag_total} "
        f"({100*diag_unlocks/diag_total:.2f}%), "
        f"off_diagonal={off_diag_unlocks}/{off_diag_total} "
        f"({100*off_diag_unlocks/off_diag_total:.2f}%)"
    )

    summary = {
        "styles": len(models),
        "calibration_locked": all(m.calibration_locked for m in models),
        "aggregate_frr": round(tot_frr, 6),
        "aggregate_far": round(tot_far, 6),
        "normal_frr": f"{tot_norm_rej}/{tot_norm_den}",
        "fatigued_frr": f"{tot_fat_rej}/{tot_fat_den}",
        "stranger_far": f"{tot_str_acc}/{tot_str_den}",
        "mimic_far": f"{tot_mim_acc}/{tot_mim_den}",
        "swap_diagonal": f"{diag_unlocks}/{diag_total}",
        "swap_off_diagonal": f"{off_diag_unlocks}/{off_diag_total}",
        "thresholds": {m.profile_id: round(m.threshold, 4) for m in models},
    }
    print("SUMMARY_JSON:", json.dumps(summary))
    return 0 if off_diag_unlocks == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run_full_benchmark())
