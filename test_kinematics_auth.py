"""test_kinematics_auth.py — Stdlib-only unittest suite for kinematics_auth & synth_traces."""
from __future__ import annotations

import random
import unittest

from kinematics_auth import (
    SessionStream,
    SymbolBurst,
    TrajectoryTick,
    calibrate_profile,
    evaluate_holdout,
    evaluate_swap_matrix,
    extract_features,
    fit_uncalibrated_profile,
    normalize_channel,
)
from synth_traces import STYLE_SPECS, _generate_session, build_profile_dataset


class KinematicsAuthTests(unittest.TestCase):
    def test_channel_normalization_and_validation(self) -> None:
        self.assertEqual(normalize_channel("direct_key"), "DIRECT_KEY")
        self.assertEqual(normalize_channel("VIRTUAL_GRID"), "VIRTUAL_GRID")
        self.assertEqual(normalize_channel("BATCH_INSERT"), "BATCH_INSERT")
        self.assertEqual(normalize_channel("batch_chunk"), "BATCH_INSERT")
        with self.assertRaises(ValueError):
            normalize_channel("UNKNOWN_CHANNEL")

    def test_hardware_polling_rate_invariance(self) -> None:
        """Downsampling an identical 110 Hz trajectory stream by 6x (~18 Hz) must preserve acceptance."""
        spec = STYLE_SPECS[0]
        ds = build_profile_dataset(spec, 9001)
        uncal = fit_uncalibrated_profile(spec.profile_id, ds.enroll_normal)
        model = calibrate_profile(
            uncal,
            ds.calib_normal + ds.calib_fatigued,
            ds.calib_stranger + ds.calib_mimic,
        )

        hi_hz = ds.holdout_normal[0]
        # Keep every 6th 'move' tick while preserving action-boundary ticks (down/up/dwell transitions)
        downsampled_ticks = []
        move_counter = 0
        for idx, tk in enumerate(hi_hz.ticks):
            is_boundary = (
                idx == 0
                or idx == len(hi_hz.ticks) - 1
                or tk.act_kind != "move"
                or hi_hz.ticks[idx - 1].act_kind != "move"
            )
            if is_boundary:
                downsampled_ticks.append(tk)
            else:
                if move_counter % 6 == 0:
                    downsampled_ticks.append(tk)
                move_counter += 1

        low_hz = SessionStream(
            session_id="downsampled_18hz",
            ticks=tuple(downsampled_ticks),
            bursts=hi_hz.bursts,
        )
        self.assertLess(len(low_hz.ticks), len(hi_hz.ticks) // 2)
        hi_res = model.score(hi_hz)
        lo_res = model.score(low_hz)
        self.assertTrue(hi_res.accepted)
        self.assertTrue(
            lo_res.accepted,
            f"Expected 6x downsampled genuine session to stay accepted, got score={lo_res.combined_score:.3f} > tau={lo_res.threshold:.3f}, top={lo_res.top_deviations}",
        )

    def test_channel_tag_invariance(self) -> None:
        """Relabeling DIRECT_KEY bursts to VIRTUAL_GRID or BATCH_INSERT must produce identical features."""
        spec = STYLE_SPECS[0]
        s_direct = _generate_session(spec, "ch_a", random.Random(42), mode="enrolled_normal")
        s_batch = SessionStream(
            session_id="ch_b",
            ticks=s_direct.ticks,
            bursts=tuple(
                SymbolBurst(
                    ts_ms=b.ts_ms,
                    channel="BATCH_INSERT",
                    symbol_count=6 if not b.is_edit else 0,
                    is_edit=b.is_edit,
                    edit_span=b.edit_span,
                )
                for b in s_direct.bursts
            ),
        )
        f1 = extract_features(s_direct)
        f2 = extract_features(s_batch)
        for a, b in zip(f1, f2):
            self.assertAlmostEqual(a, b, places=9)

    def test_pre_test_calibration_lock_and_no_double_calibration(self) -> None:
        ds = build_profile_dataset(STYLE_SPECS[1], 12345)
        uncal = fit_uncalibrated_profile(ds.spec.profile_id, ds.enroll_normal)
        self.assertFalse(uncal.calibration_locked)
        with self.assertRaises(ValueError):
            evaluate_holdout(
                uncal,
                ds.holdout_normal,
                ds.holdout_fatigued,
                ds.holdout_stranger,
                ds.holdout_mimic,
            )

        locked = calibrate_profile(
            uncal,
            ds.calib_normal + ds.calib_fatigued,
            ds.calib_stranger + ds.calib_mimic,
        )
        self.assertTrue(locked.calibration_locked)
        with self.assertRaises(ValueError):
            calibrate_profile(locked, ds.calib_normal, ds.calib_stranger)

    def test_sequential_cusum_detects_mid_session_takeover(self) -> None:
        """First half genuine + second half stranger hijack must trigger higher sequential anomaly."""
        spec = STYLE_SPECS[0]
        ds = build_profile_dataset(spec, 5555)
        uncal = fit_uncalibrated_profile(spec.profile_id, ds.enroll_normal)
        model = calibrate_profile(
            uncal,
            ds.calib_normal + ds.calib_fatigued,
            ds.calib_stranger + ds.calib_mimic,
        )

        genuine = ds.holdout_normal[0]
        hijacker = ds.holdout_mimic[0]

        g_t_cut = genuine.ticks[len(genuine.ticks) // 2].ts_ms
        g_b_cut = genuine.bursts[len(genuine.bursts) // 2].ts_ms
        cut = max(g_t_cut, g_b_cut)

        first_ticks = [t for t in genuine.ticks if t.ts_ms <= cut]
        first_bursts = [b for b in genuine.bursts if b.ts_ms <= cut]

        h_t0 = hijacker.ticks[0].ts_ms
        h_b0 = hijacker.bursts[0].ts_ms
        shift = (cut + 300.0) - min(h_t0, h_b0)
        second_ticks = [
            TrajectoryTick(t.ts_ms + shift, t.x_norm, t.y_norm, t.act_kind)
            for t in hijacker.ticks[len(hijacker.ticks) // 2 :]
        ]
        second_bursts = [
            SymbolBurst(b.ts_ms + shift, b.channel, b.symbol_count, b.is_edit, b.edit_span)
            for b in hijacker.bursts[len(hijacker.bursts) // 2 :]
        ]
        hybrid = SessionStream(
            session_id="mid_session_takeover",
            ticks=tuple(first_ticks + second_ticks),
            bursts=tuple(first_bursts + second_bursts),
        )

        g_res = model.score(genuine)
        h_res = model.score(hybrid)
        self.assertTrue(g_res.accepted)
        self.assertGreater(h_res.sequential_cusum, g_res.sequential_cusum)
        self.assertFalse(h_res.accepted)

    def test_full_benchmark_far_frr_and_zero_cross_profile_unlocks(self) -> None:
        datasets = [
            build_profile_dataset(spec, 20261125 + idx * 10_000)
            for idx, spec in enumerate(STYLE_SPECS)
        ]
        models = []
        genuine_map = {}
        for ds in datasets:
            uncal = fit_uncalibrated_profile(ds.spec.profile_id, ds.enroll_normal)
            model = calibrate_profile(
                uncal,
                ds.calib_normal + ds.calib_fatigued,
                ds.calib_stranger + ds.calib_mimic,
            )
            models.append(model)
            rep = evaluate_holdout(
                model,
                ds.holdout_normal,
                ds.holdout_fatigued,
                ds.holdout_stranger,
                ds.holdout_mimic,
            )
            self.assertLessEqual(rep.total_frr, 0.12, f"High FRR on {rep.profile_id}: {rep.total_frr}")
            self.assertLessEqual(rep.total_far, 0.08, f"High FAR on {rep.profile_id}: {rep.total_far}")
            genuine_map[ds.spec.profile_id] = ds.holdout_normal + ds.holdout_fatigued

        swap = evaluate_swap_matrix(models, genuine_map)
        for m_a in models:
            for m_b in models:
                acc, tot = swap[m_a.profile_id][m_b.profile_id]
                if m_a.profile_id == m_b.profile_id:
                    self.assertGreaterEqual(acc / tot, 0.85)
                else:
                    self.assertEqual(
                        acc,
                        0,
                        f"Cross-profile unlock detected: {m_b.profile_id} unlocked {m_a.profile_id} ({acc}/{tot})",
                    )


if __name__ == "__main__":
    unittest.main()
