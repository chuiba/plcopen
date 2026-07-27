"""Tests for the SA5 LeRobot dataset replay bridge (tools/lerobot_replay.py).

Three layers, matching the tool's dependency discipline:
- standard-library layer (loaders, cycle mapping, metrics): always runs;
- parquet layer: self-skips unless pyarrow (or pandas) is importable;
- sim layer (JointStreamSim replay): self-skips unless pyplcopen is
  importable; exercised by twin CI.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import sys

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

import lerobot_replay as bridge  # noqa: E402

FIXTURE = TOOLS_DIR / "fixtures" / "so_arm_synthetic_30fps.csv"


def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


class CsvLoaderTests(unittest.TestCase):
    def test_fixture_loads_with_inferred_fps(self) -> None:
        episode = bridge.load_csv(FIXTURE)
        self.assertEqual(len(episode.frames), 91)
        self.assertEqual(episode.joint_count, 6)
        self.assertAlmostEqual(episode.fps, 30.0, delta=0.2)
        self.assertAlmostEqual(episode.duration, 3.0, delta=1e-6)
        self.assertEqual(episode.joint_names[0], "joint1")
        # t=0 golden: 0.35*sin(0.7j) + 0.12*sin(0); joint1 (j=0) is exactly 0.
        self.assertAlmostEqual(episode.frames[0][0], 0.0, delta=1e-9)
        self.assertAlmostEqual(
            episode.frames[0][1], 0.35 * math.sin(0.7), delta=2e-6
        )

    def test_headerless_csv_needs_fps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plain.csv"
            path.write_text("0.0,0.1\n0.2,0.3\n0.4,0.5\n")
            with self.assertRaisesRegex(bridge.ReplayError, "--fps"):
                bridge.load_csv(path)
            episode = bridge.load_csv(path, fps=50.0)
            self.assertEqual(episode.joint_count, 2)
            self.assertEqual(episode.timestamps, [0.0, 0.02, 0.04])
            self.assertEqual(episode.joint_names, ["joint1", "joint2"])

    def test_bom_header_is_recognized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bom.csv"
            path.write_bytes(b"\xef\xbb\xbf" + FIXTURE.read_bytes())
            episode = bridge.load_csv(path)
            self.assertEqual(episode.joint_count, 6)
            self.assertEqual(episode.joint_names[0], "joint1")

    def test_zero_fps_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plain.csv"
            path.write_text("0.0,0.1\n0.2,0.3\n0.4,0.5\n")
            with self.assertRaisesRegex(bridge.ReplayError, "positive"):
                bridge.load_csv(path, fps=0.0)
            self.assertEqual(bridge.main(["info", str(path), "--fps", "0"]), 2)

    def test_malformed_csv_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ragged = Path(directory) / "ragged.csv"
            ragged.write_text("timestamp,a,b\n0.0,1.0,2.0\n0.1,1.0\n")
            with self.assertRaises(bridge.ReplayError):
                bridge.load_csv(ragged)
            backwards = Path(directory) / "backwards.csv"
            backwards.write_text("timestamp,a\n0.1,1.0\n0.0,2.0\n")
            with self.assertRaisesRegex(bridge.ReplayError, "increasing"):
                bridge.load_csv(backwards)
            single = Path(directory) / "single.csv"
            single.write_text("timestamp,a\n0.0,1.0\n")
            with self.assertRaises(bridge.ReplayError):
                bridge.load_csv(single)


class JsonLoaderTests(unittest.TestCase):
    def test_json_with_fps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.json"
            path.write_text(
                json.dumps(
                    {
                        "fps": 25.0,
                        "joint_names": ["pan", "lift"],
                        "frames": [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]],
                    }
                )
            )
            episode = bridge.load_json(path)
            self.assertEqual(episode.fps, 25.0)
            self.assertEqual(episode.joint_names, ["pan", "lift"])
            self.assertEqual(episode.timestamps, [0.0, 0.04, 0.08])

    def test_json_infers_fps_from_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.json"
            path.write_text(
                json.dumps(
                    {
                        "timestamps": [0.0, 0.1, 0.2],
                        "frames": [[0.0], [1.0], [2.0]],
                    }
                )
            )
            episode = bridge.load_json(path)
            self.assertAlmostEqual(episode.fps, 10.0, delta=1e-9)

    def test_json_without_time_base_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.json"
            path.write_text(json.dumps({"frames": [[0.0], [1.0]]}))
            with self.assertRaisesRegex(bridge.ReplayError, "fps"):
                bridge.load_json(path)

    def test_json_malformed_shapes_raise_replay_error(self) -> None:
        cases = [
            {"frames": 5, "fps": 30},
            {"frames": [[1.0], [2.0]], "timestamps": [0.0, "x"]},
            {"frames": [[1.0], [2.0]], "fps": "abc"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, payload in enumerate(cases):
                path = Path(directory) / f"bad{index}.json"
                path.write_text(json.dumps(payload))
                with self.assertRaises(bridge.ReplayError, msg=str(payload)):
                    bridge.load_json(path)


class CycleFrameTests(unittest.TestCase):
    def test_fixture_maps_to_1khz_cycles(self) -> None:
        episode = bridge.load_csv(FIXTURE)
        frames = bridge.to_cycle_frames(episode)
        self.assertEqual(frames.stamps[0], 1)
        # t = 1/30 s at 1 kHz rounds to cycle 33, shifted to first-frame base 1.
        self.assertEqual(frames.stamps[1], 34)
        self.assertEqual(frames.frame_interval, 33)
        self.assertEqual(len(frames.stamps), 91)
        self.assertTrue(
            all(b > a for a, b in zip(frames.stamps, frames.stamps[1:]))
        )

    def test_coarse_cycle_rate_stays_strictly_increasing(self) -> None:
        episode = bridge.load_csv(FIXTURE)
        frames = bridge.to_cycle_frames(episode, cycle_hz=10.0)
        self.assertTrue(
            all(b > a for a, b in zip(frames.stamps, frames.stamps[1:]))
        )
        # The bump count exposes that this mapping stretched the time base
        # (30 fps into 10 Hz: every frame after the first collides).
        self.assertEqual(frames.bumped_frames, 90)
        fine = bridge.to_cycle_frames(episode)
        self.assertEqual(fine.bumped_frames, 0)

    def test_scale_offset_and_joint_count(self) -> None:
        episode = bridge.load_csv(FIXTURE)
        frames = bridge.to_cycle_frames(
            episode, scale=[2.0], offset=[0.5], joint_count=2
        )
        self.assertEqual(frames.joint_count, 2)
        self.assertAlmostEqual(
            frames.positions[0][1],
            episode.frames[0][1] * 2.0 + 0.5,
            delta=1e-12,
        )
        per_joint = bridge.to_cycle_frames(
            episode, scale=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        self.assertAlmostEqual(
            per_joint.positions[0][5], episode.frames[0][5] * 6.0, delta=1e-12
        )
        with self.assertRaisesRegex(bridge.ReplayError, "scale"):
            bridge.to_cycle_frames(episode, scale=[1.0, 2.0])
        with self.assertRaises(bridge.ReplayError):
            bridge.to_cycle_frames(episode, joint_count=7)


class MetricsTests(unittest.TestCase):
    def test_third_difference_of_cubic_is_constant(self) -> None:
        samples = [float(i**3) for i in range(32)]
        stats = bridge.third_difference_stats(samples)
        self.assertAlmostEqual(stats.max_abs, 6.0, delta=1e-9)
        self.assertAlmostEqual(stats.rms, 6.0, delta=1e-9)
        flat = bridge.third_difference_stats([1.0] * 16)
        self.assertEqual(flat.max_abs, 0.0)

    def test_zoh_and_linear_reference(self) -> None:
        frames = bridge.CycleFrames(
            cycle_hz=1000.0,
            stamps=[1, 5],
            positions=[[0.0], [1.0]],
            velocities=[[0.25], [0.0]],
            frame_interval=4,
        )
        self.assertEqual(
            bridge.zoh_track(frames, 6, 0), [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        )
        reference = bridge.linear_reference(frames, 6, 0)
        self.assertEqual(reference[0], 0.0)
        self.assertAlmostEqual(reference[2], 0.5, delta=1e-12)
        self.assertEqual(reference[4], 1.0)
        self.assertEqual(reference[5], 1.0)


class DatasetRootTests(unittest.TestCase):
    def test_lerobot_root_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "fps": 30,
                        "chunks_size": 1000,
                        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                    }
                )
            )
            episode_file = root / "data" / "chunk-000" / "episode_000002.parquet"
            episode_file.parent.mkdir(parents=True)
            episode_file.write_bytes(b"")
            resolved, fps = bridge.resolve_lerobot_episode_file(root, 2)
            self.assertEqual(resolved, episode_file.resolve())
            self.assertEqual(fps, 30.0)
            with self.assertRaisesRegex(bridge.ReplayError, "not found"):
                bridge.resolve_lerobot_episode_file(root, 3)

    def test_non_dataset_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(bridge.ReplayError, "info.json"):
                bridge.resolve_lerobot_episode_file(Path(directory), 0)

    def _write_info(self, root: Path, payload: dict) -> None:
        (root / "meta").mkdir(exist_ok=True)
        (root / "meta" / "info.json").write_text(json.dumps(payload))

    def test_untrusted_data_path_is_contained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Absolute template: must not escape the dataset root.
            self._write_info(root, {"fps": 30, "data_path": "/etc/hostname"})
            with self.assertRaisesRegex(bridge.ReplayError, "relative"):
                bridge.resolve_lerobot_episode_file(root, 0)
            # Parent-directory traversal: same containment rule.
            self._write_info(
                root, {"fps": 30, "data_path": "../outside_{episode_index}.parquet"}
            )
            with self.assertRaisesRegex(bridge.ReplayError, "escapes"):
                bridge.resolve_lerobot_episode_file(root, 0)

    def test_non_v2_template_and_bad_chunks_size_are_replay_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # LeRobot v3 layout uses {chunk_index}/{file_index}: clean error,
            # not a raw KeyError.
            self._write_info(
                root,
                {
                    "fps": 30,
                    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                },
            )
            with self.assertRaisesRegex(bridge.ReplayError, "data_path template"):
                bridge.resolve_lerobot_episode_file(root, 0)
            self._write_info(
                root,
                {"fps": 30, "chunks_size": 0, "data_path": "data/e_{episode_index}.parquet"},
            )
            with self.assertRaisesRegex(bridge.ReplayError, "chunks_size"):
                bridge.resolve_lerobot_episode_file(root, 0)


@unittest.skipUnless(
    _has_module("pyarrow") or _has_module("pandas"),
    "parquet layer needs pyarrow or pandas (optional dependency)",
)
class ParquetLoaderTests(unittest.TestCase):
    def test_parquet_roundtrip(self) -> None:
        pyarrow = None
        try:
            import pyarrow  # type: ignore
            import pyarrow.parquet  # noqa: F401
        except ImportError:
            self.skipTest("writing the test parquet file needs pyarrow")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_000000.parquet"
            table = pyarrow.table(
                {
                    "action": [[0.0, 0.5], [0.1, 0.6], [0.2, 0.7]],
                    "timestamp": [0.0, 1.0 / 30.0, 2.0 / 30.0],
                }
            )
            pyarrow.parquet.write_table(table, path)
            episode = bridge.load_parquet(path)
            self.assertEqual(episode.joint_count, 2)
            self.assertAlmostEqual(episode.fps, 30.0, delta=0.2)
            self.assertAlmostEqual(episode.frames[2][1], 0.7, delta=1e-12)
            with self.assertRaisesRegex(bridge.ReplayError, "column 'state'"):
                bridge.load_parquet(path, column="state")


@unittest.skipUnless(_has_module("pyplcopen"), "sim layer needs pyplcopen")
class ReplaySimTests(unittest.TestCase):
    def test_fixture_replay_ab_metrics(self) -> None:
        episode = bridge.load_csv(FIXTURE)
        frames = bridge.to_cycle_frames(episode)
        result = bridge.replay_sim(frames)

        self.assertEqual(result.joint_count, 6)
        self.assertEqual(result.pushed_frames, 91)
        self.assertEqual(result.rejected_frames, 0)
        self.assertEqual(result.dropouts, 0)
        self.assertEqual(result.tail_dropouts, 0)
        for metrics in result.joints:
            # Construction promise: the filtered stream respects the
            # configured per-cycle jerk limit (0.02) with headroom.
            self.assertLessEqual(metrics.filtered.max_abs, 0.022)
            # Recorded-baseline ratchets (measured on this fixture through
            # the binding: separation 10.4-11.1x, tracking <= 0.0681,
            # final_error <= 0.0301 — the binding's always-on position
            # envelope costs jerk headroom versus the SA4 C++ numbers; see
            # the SA5 commit evidence). Generous margins so only real
            # regressions trip.
            self.assertGreaterEqual(metrics.separation, 5.0)
            self.assertLessEqual(metrics.tracking_error, 0.12)
            self.assertLessEqual(metrics.final_error, 0.06)

    def test_replay_sim_cli_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            code = bridge.main(
                [
                    "replay-sim",
                    str(FIXTURE),
                    "--jerk-gate",
                    "0.022",
                    "--report",
                    str(report),
                ]
            )
            self.assertEqual(code, 0)
            payload = json.loads(report.read_text())
            self.assertEqual(payload["tool"], "lerobot_replay")
            self.assertEqual(payload["failures"], [])
            self.assertEqual(payload["result"]["rejected_frames"], 0)
            self.assertEqual(payload["bumped_frames"], 0)
            self.assertEqual(len(payload["result"]["joints"]), 6)

    def test_out_of_envelope_mapping_is_a_clean_error(self) -> None:
        # Fixture max |q| is ~0.47; scale 100 exceeds the default ±pi
        # envelope of the sim facade. Must exit 2 (usage error), not a
        # traceback, and must point at the mapping flags.
        code = bridge.main(["replay-sim", str(FIXTURE), "--scale", "100"])
        self.assertEqual(code, 2)

    def test_position_limit_flag_admits_scaled_units(self) -> None:
        episode = bridge.load_csv(FIXTURE)
        frames = bridge.to_cycle_frames(episode, scale=[100.0], joint_count=1)
        result = bridge.replay_sim(
            frames,
            velocity_limit=80.0,
            acceleration_limit=8.0,
            jerk_limit=2.0,
            position_limit=100.0,
        )
        self.assertEqual(result.rejected_frames, 0)
        self.assertEqual(result.dropouts, 0)

    def test_short_episode_tracking_is_measured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.csv"
            path.write_text(
                "timestamp,j1\n0.000000,0.0\n0.033333,0.3\n"
                "0.066667,0.6\n0.100000,0.9\n"
            )
            frames = bridge.to_cycle_frames(bridge.load_csv(path))
            result = bridge.replay_sim(frames)
            # An empty window must not report a perfect 0.0.
            self.assertGreater(result.joints[0].tracking_error, 0.0)

    def test_constant_episode_report_stays_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "const.csv"
            rows = ["timestamp,j1,j2"] + [
                f"{i / 30.0:.6f},0.100000,-0.200000" for i in range(30)
            ]
            path.write_text("\n".join(rows) + "\n")
            report = Path(directory) / "report.json"
            code = bridge.main(
                ["replay-sim", str(path), "--report", str(report)]
            )
            self.assertEqual(code, 0)
            text = report.read_text()
            self.assertNotIn("Infinity", text)
            payload = json.loads(text)
            self.assertIsNone(payload["result"]["joints"][0]["separation"])

    def test_missing_report_parent_fails_before_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "no-such-dir" / "report.json"
            code = bridge.main(
                ["replay-sim", str(FIXTURE), "--report", str(missing)]
            )
            self.assertEqual(code, 2)


class CliInfoTests(unittest.TestCase):
    def test_info_runs_on_fixture(self) -> None:
        self.assertEqual(bridge.main(["info", str(FIXTURE)]), 0)

    def test_unsupported_format_reports_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.txt"
            path.write_text("nope")
            self.assertEqual(bridge.main(["info", str(path)]), 2)


if __name__ == "__main__":
    unittest.main()
