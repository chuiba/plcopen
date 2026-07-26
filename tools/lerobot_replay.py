#!/usr/bin/env python3
"""LeRobot dataset replay bridge (S5 batch SA5).

Bridges a recorded low-rate intent stream (LeRobot episode: teleoperation or
policy output, typically 30 fps) into the plcopen stream layer: dataset
frames -> cycle-domain command frames (KB-035 contract: strictly increasing
producer timestamps) -> JointStreamSim upsample replay with a zero-order-hold
A/B comparison using the SA4 third-difference jerk metrics.

Layering (mirrors the SA1 instrument discipline):
- Episode loading and all metric math are standard-library only. LeRobot v2
  parquet reading is an OPTIONAL dependency path (pyarrow, else pandas),
  imported only when a parquet source is opened; CSV/JSON loaders cover the
  dependency-free case and CI.
- Unit mapping is EXPLICIT (--scale/--offset). The 4.8 gate of the approved
  Feetech matrix keeps physical unit scales unverified, so this tool bakes in
  no unit constants whatsoever; the caller states the dataset-to-radian (or
  any other) mapping in full.
- No dataset is vendored: the repository ships only a synthetic fixture
  (tools/fixtures/so_arm_synthetic_30fps.csv); real datasets are local paths
  provided by the operator (same discipline as KB-093 twin models).

The replay-sim mode needs the pyplcopen binding (JointStreamSim) and is
exercised by twin CI; without the binding the CLI reports the missing
dependency and the tests self-skip.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_CYCLE_HZ = 1000.0
DEFAULT_PARQUET_COLUMN = "action"


class ReplayError(ValueError):
    """Raised for malformed episodes or invalid bridge parameters."""


# --- episode model ----------------------------------------------------------


@dataclasses.dataclass
class Episode:
    fps: float
    timestamps: List[float]  # seconds, strictly increasing
    frames: List[List[float]]  # one row per frame, one column per joint
    joint_names: List[str]
    source: str

    @property
    def joint_count(self) -> int:
        return len(self.frames[0]) if self.frames else 0

    @property
    def duration(self) -> float:
        return self.timestamps[-1] - self.timestamps[0] if self.timestamps else 0.0


def _validate_episode(episode: Episode) -> Episode:
    if len(episode.frames) < 2:
        raise ReplayError("episode needs at least two frames")
    width = len(episode.frames[0])
    if width < 1:
        raise ReplayError("episode frames are empty")
    if len(episode.timestamps) != len(episode.frames):
        raise ReplayError("timestamp count does not match frame count")
    for row in episode.frames:
        if len(row) != width:
            raise ReplayError("ragged frame rows")
        for value in row:
            if not math.isfinite(value):
                raise ReplayError("non-finite frame value")
    previous = None
    for stamp in episode.timestamps:
        if not math.isfinite(stamp):
            raise ReplayError("non-finite timestamp")
        if previous is not None and stamp <= previous:
            raise ReplayError("timestamps must be strictly increasing")
        previous = stamp
    if not (math.isfinite(episode.fps) and episode.fps > 0.0):
        raise ReplayError("fps must be positive and finite")
    if len(episode.joint_names) != width:
        raise ReplayError("joint name count does not match frame width")
    return episode


def _infer_fps(timestamps: Sequence[float]) -> float:
    deltas = sorted(
        timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
    )
    if not deltas:
        raise ReplayError("cannot infer fps from fewer than two timestamps")
    median = deltas[len(deltas) // 2]
    if median <= 0.0:
        raise ReplayError("cannot infer fps: non-increasing timestamps")
    return 1.0 / median


# --- standard-library loaders (CSV / JSON) ----------------------------------


def load_csv(path: Path, fps: Optional[float] = None) -> Episode:
    with open(path, newline="") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if not rows:
        raise ReplayError(f"{path}: empty CSV")

    def numeric(cell: str) -> bool:
        try:
            float(cell)
            return True
        except ValueError:
            return False

    header: Optional[List[str]] = None
    if not all(numeric(cell) for cell in rows[0]):
        header = [cell.strip() for cell in rows[0]]
        rows = rows[1:]
    if not rows:
        raise ReplayError(f"{path}: CSV has a header but no data rows")

    timestamp_column: Optional[int] = None
    if header is not None:
        lowered = [name.lower() for name in header]
        if "timestamp" in lowered:
            timestamp_column = lowered.index("timestamp")

    timestamps: List[float] = []
    frames: List[List[float]] = []
    for row in rows:
        try:
            values = [float(cell) for cell in row]
        except ValueError as error:
            raise ReplayError(f"{path}: non-numeric cell ({error})") from None
        if timestamp_column is not None:
            timestamps.append(values.pop(timestamp_column))
        frames.append(values)

    if timestamp_column is None:
        if fps is None:
            raise ReplayError(
                f"{path}: no timestamp column; pass --fps to synthesize timestamps"
            )
        timestamps = [index / fps for index in range(len(frames))]
    elif fps is None:
        fps = _infer_fps(timestamps)

    if header is not None:
        names = [
            name
            for index, name in enumerate(header)
            if index != timestamp_column
        ]
    else:
        names = [f"joint{index + 1}" for index in range(len(frames[0]))]

    return _validate_episode(
        Episode(
            fps=fps,
            timestamps=timestamps,
            frames=frames,
            joint_names=names,
            source=str(path),
        )
    )


def load_json(path: Path, fps: Optional[float] = None) -> Episode:
    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "frames" not in payload:
        raise ReplayError(f"{path}: expected an object with a 'frames' array")
    frames = [[float(value) for value in row] for row in payload["frames"]]
    if fps is None:
        fps = payload.get("fps")
    timestamps = payload.get("timestamps")
    if timestamps is not None:
        timestamps = [float(value) for value in timestamps]
        if fps is None:
            fps = _infer_fps(timestamps)
    else:
        if fps is None:
            raise ReplayError(f"{path}: needs 'fps' or 'timestamps'")
        timestamps = [index / float(fps) for index in range(len(frames))]
    names = payload.get("joint_names")
    if names is None:
        names = [f"joint{index + 1}" for index in range(len(frames[0]))] if frames else []
    return _validate_episode(
        Episode(
            fps=float(fps),
            timestamps=timestamps,
            frames=frames,
            joint_names=[str(name) for name in names],
            source=str(path),
        )
    )


# --- optional-dependency loader (LeRobot v2 parquet) ------------------------


def _parquet_records(path: Path) -> List[Dict[str, object]]:
    try:
        import pyarrow.parquet  # type: ignore

        return pyarrow.parquet.read_table(path).to_pylist()
    except ImportError:
        pass
    try:
        import pandas  # type: ignore

        return pandas.read_parquet(path).to_dict("records")
    except ImportError:
        raise ReplayError(
            f"{path}: reading parquet needs pyarrow or pandas "
            "(optional dependency; use CSV/JSON for the dependency-free path)"
        ) from None


def load_parquet(
    path: Path,
    column: str = DEFAULT_PARQUET_COLUMN,
    fps: Optional[float] = None,
) -> Episode:
    records = _parquet_records(path)
    if not records:
        raise ReplayError(f"{path}: empty parquet table")
    if column not in records[0]:
        available = ", ".join(sorted(str(key) for key in records[0]))
        raise ReplayError(f"{path}: column '{column}' not found (has: {available})")

    frames = [[float(value) for value in record[column]] for record in records]
    if "timestamp" in records[0]:
        timestamps = [float(record["timestamp"]) for record in records]
        if fps is None:
            fps = _infer_fps(timestamps)
    else:
        if fps is None:
            raise ReplayError(f"{path}: no 'timestamp' column; pass --fps")
        timestamps = [index / fps for index in range(len(frames))]
    names = [f"{column}[{index}]" for index in range(len(frames[0]))] if frames else []
    return _validate_episode(
        Episode(
            fps=fps,
            timestamps=timestamps,
            frames=frames,
            joint_names=names,
            source=str(path),
        )
    )


def resolve_lerobot_episode_file(root: Path, episode: int) -> "tuple[Path, float]":
    """Resolve data file + fps from a LeRobot v2 dataset root (meta/info.json)."""
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ReplayError(f"{root}: not a LeRobot v2 dataset root (no meta/info.json)")
    with open(info_path) as handle:
        info = json.load(handle)
    try:
        fps = float(info["fps"])
        data_path = str(info["data_path"])
        chunks_size = int(info.get("chunks_size", 1000))
    except (KeyError, TypeError, ValueError) as error:
        raise ReplayError(f"{info_path}: malformed info.json ({error})") from None
    relative = data_path.format(
        episode_chunk=episode // chunks_size, episode_index=episode
    )
    episode_path = root / relative
    if not episode_path.is_file():
        raise ReplayError(f"{episode_path}: episode file not found")
    return episode_path, fps


def load_episode(
    path: Path,
    fps: Optional[float] = None,
    column: str = DEFAULT_PARQUET_COLUMN,
    episode: int = 0,
) -> Episode:
    path = Path(path)
    if path.is_dir():
        episode_path, dataset_fps = resolve_lerobot_episode_file(path, episode)
        return load_parquet(episode_path, column=column, fps=fps or dataset_fps)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path, fps=fps)
    if suffix == ".json":
        return load_json(path, fps=fps)
    if suffix == ".parquet":
        return load_parquet(path, column=column, fps=fps)
    raise ReplayError(f"{path}: unsupported episode format '{suffix}'")


# --- cycle-domain conversion ------------------------------------------------


@dataclasses.dataclass
class CycleFrames:
    cycle_hz: float
    stamps: List[int]  # strictly increasing cycle indices, first >= 1
    positions: List[List[float]]
    velocities: List[List[float]]  # per-cycle units, finite-difference ff
    frame_interval: int  # median stamp spacing, cycles

    @property
    def joint_count(self) -> int:
        return len(self.positions[0]) if self.positions else 0


def _broadcast(name: str, values: Optional[Sequence[float]], width: int, default: float) -> List[float]:
    if values is None:
        return [default] * width
    if len(values) == 1:
        return [float(values[0])] * width
    if len(values) != width:
        raise ReplayError(
            f"{name} needs 1 value or one per joint ({width}), got {len(values)}"
        )
    return [float(value) for value in values]


def to_cycle_frames(
    episode: Episode,
    cycle_hz: float = DEFAULT_CYCLE_HZ,
    scale: Optional[Sequence[float]] = None,
    offset: Optional[Sequence[float]] = None,
    joint_count: Optional[int] = None,
) -> CycleFrames:
    if not (math.isfinite(cycle_hz) and cycle_hz > 0.0):
        raise ReplayError("cycle_hz must be positive and finite")
    width = episode.joint_count
    if joint_count is not None:
        if joint_count < 1 or joint_count > width:
            raise ReplayError(
                f"joint_count {joint_count} outside 1..{width} (episode width)"
            )
        width = joint_count
    scales = _broadcast("scale", scale, width, 1.0)
    offsets = _broadcast("offset", offset, width, 0.0)

    base = episode.timestamps[0]
    stamps: List[int] = []
    positions: List[List[float]] = []
    previous = 0
    for stamp_s, row in zip(episode.timestamps, episode.frames):
        cycle = int(round((stamp_s - base) * cycle_hz)) + 1
        if cycle <= previous:  # keep the KB-035 strictly-increasing contract
            cycle = previous + 1
        previous = cycle
        stamps.append(cycle)
        positions.append(
            [row[joint] * scales[joint] + offsets[joint] for joint in range(width)]
        )

    # Velocity feedforward (per-cycle units) from forward differences of the
    # frames themselves — derived data, mirroring what SA4 showed the
    # upsample filter needs to track a moving intent instead of braking to a
    # standstill at every frame target. The final frame gets zero velocity.
    velocities: List[List[float]] = []
    for i in range(len(stamps)):
        if i + 1 < len(stamps):
            span = float(stamps[i + 1] - stamps[i])
            velocities.append(
                [
                    (positions[i + 1][joint] - positions[i][joint]) / span
                    for joint in range(width)
                ]
            )
        else:
            velocities.append([0.0] * width)

    deltas = sorted(stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1))
    interval = deltas[len(deltas) // 2] if deltas else 1
    return CycleFrames(
        cycle_hz=cycle_hz,
        stamps=stamps,
        positions=positions,
        velocities=velocities,
        frame_interval=max(1, interval),
    )


# --- metrics (SA4 third-difference shape) -----------------------------------


@dataclasses.dataclass
class JerkStats:
    max_abs: float
    rms: float


def third_difference_stats(samples: Sequence[float]) -> JerkStats:
    max_abs = 0.0
    sum_sq = 0.0
    count = 0
    for i in range(3, len(samples)):
        jerk = (
            samples[i]
            - 3.0 * samples[i - 1]
            + 3.0 * samples[i - 2]
            - samples[i - 3]
        )
        magnitude = abs(jerk)
        if magnitude > max_abs:
            max_abs = magnitude
        sum_sq += jerk * jerk
        count += 1
    rms = math.sqrt(sum_sq / count) if count else 0.0
    return JerkStats(max_abs=max_abs, rms=rms)


def zoh_track(frames: CycleFrames, total_cycles: int, joint: int) -> List[float]:
    """Baseline A: hold the newest frame value at every command cycle."""
    track: List[float] = []
    value = frames.positions[0][joint]
    index = 0
    for cycle in range(1, total_cycles + 1):
        while index < len(frames.stamps) and frames.stamps[index] <= cycle:
            value = frames.positions[index][joint]
            index += 1
        track.append(value)
    return track


def linear_reference(frames: CycleFrames, total_cycles: int, joint: int) -> List[float]:
    """Piecewise-linear interpolation of the frames, held flat at both ends."""
    track: List[float] = []
    index = 0
    for cycle in range(1, total_cycles + 1):
        while (
            index + 1 < len(frames.stamps) and frames.stamps[index + 1] <= cycle
        ):
            index += 1
        if cycle <= frames.stamps[0]:
            track.append(frames.positions[0][joint])
        elif index + 1 >= len(frames.stamps):
            track.append(frames.positions[-1][joint])
        else:
            left = frames.stamps[index]
            right = frames.stamps[index + 1]
            blend = (cycle - left) / (right - left)
            track.append(
                frames.positions[index][joint] * (1.0 - blend)
                + frames.positions[index + 1][joint] * blend
            )
    return track


# --- JointStreamSim replay (optional pyplcopen) -----------------------------


@dataclasses.dataclass
class JointReplayMetrics:
    zoh: JerkStats
    filtered: JerkStats
    separation: float
    tracking_error: float
    final_error: float


@dataclasses.dataclass
class ReplayResult:
    joint_count: int
    total_cycles: int
    pushed_frames: int
    rejected_frames: int
    dropouts: int  # during the episode (a real delivery defect)
    tail_dropouts: int  # after the last frame: the watchdog ladder, by design
    joints: List[JointReplayMetrics]

    def as_dict(self) -> Dict[str, object]:
        return {
            "joint_count": self.joint_count,
            "total_cycles": self.total_cycles,
            "pushed_frames": self.pushed_frames,
            "rejected_frames": self.rejected_frames,
            "dropouts": self.dropouts,
            "tail_dropouts": self.tail_dropouts,
            "joints": [
                {
                    "zoh_max_jerk": joint.zoh.max_abs,
                    "zoh_rms_jerk": joint.zoh.rms,
                    "filtered_max_jerk": joint.filtered.max_abs,
                    "filtered_rms_jerk": joint.filtered.rms,
                    "separation": joint.separation,
                    "tracking_error": joint.tracking_error,
                    "final_error": joint.final_error,
                }
                for joint in self.joints
            ],
        }


def replay_sim(
    frames: CycleFrames,
    mode: str = "upsample",
    velocity_limit: float = 0.8,
    acceleration_limit: float = 0.08,
    jerk_limit: float = 0.02,
    timeout_cycles: Optional[int] = None,
    extrapolation_cycles: Optional[int] = None,
    settle_cycles: Optional[int] = None,
    position_limit: float = math.pi,
) -> ReplayResult:
    try:
        import pyplcopen  # type: ignore
    except ImportError:
        raise ReplayError(
            "replay-sim needs the pyplcopen binding (pip install .)"
        ) from None

    interval = frames.frame_interval
    if timeout_cycles is None:
        timeout_cycles = 4 * interval
    if extrapolation_cycles is None:
        extrapolation_cycles = 2 * interval
    if settle_cycles is None:
        settle_cycles = 2 * interval
    total_cycles = frames.stamps[-1] + settle_cycles

    stream = pyplcopen.JointStreamSim(
        frames.joint_count,
        mode,
        velocity_limit,
        acceleration_limit,
        jerk_limit,
        timeout_cycles,
        extrapolation_cycles,
        position_limit,
    )
    stream.reset(list(frames.positions[0]))

    emitted: List[List[float]] = [[] for _ in range(frames.joint_count)]
    index = 0
    pushed = 0
    episode_dropouts = 0
    for cycle in range(1, total_cycles + 1):
        if index < len(frames.stamps) and frames.stamps[index] == cycle:
            stream.push_frame(
                list(frames.positions[index]),
                cycle,
                list(frames.velocities[index]),
            )
            index += 1
            pushed += 1
            if index == len(frames.stamps):
                episode_dropouts = stream.dropouts()
        stream.cycle()
        snapshot = stream.setpoint_frame()
        for joint in range(frames.joint_count):
            emitted[joint].append(snapshot["positions"][joint])

    joints: List[JointReplayMetrics] = []
    skip = min(4 * interval, total_cycles)  # let the filter converge first
    # Tracking is judged only while frames keep arriving; the tail after the
    # last frame belongs to final_error (and, past timeout_cycles, to the
    # watchdog dropout ladder by design).
    tracking_end = min(frames.stamps[-1], total_cycles)
    for joint in range(frames.joint_count):
        baseline = zoh_track(frames, total_cycles, joint)
        reference = linear_reference(frames, total_cycles, joint)
        zoh_stats = third_difference_stats(baseline)
        filtered_stats = third_difference_stats(emitted[joint])
        tracking = 0.0
        for cycle in range(skip, tracking_end):
            error = abs(emitted[joint][cycle] - reference[cycle])
            if error > tracking:
                tracking = error
        joints.append(
            JointReplayMetrics(
                zoh=zoh_stats,
                filtered=filtered_stats,
                separation=(
                    zoh_stats.max_abs / filtered_stats.max_abs
                    if filtered_stats.max_abs > 0.0
                    else math.inf
                ),
                tracking_error=tracking,
                final_error=abs(emitted[joint][-1] - frames.positions[-1][joint]),
            )
        )

    total_dropouts = stream.dropouts()
    return ReplayResult(
        joint_count=frames.joint_count,
        total_cycles=total_cycles,
        pushed_frames=pushed,
        rejected_frames=stream.rejected_frames(),
        dropouts=episode_dropouts,
        tail_dropouts=total_dropouts - episode_dropouts,
        joints=joints,
    )


# --- CLI --------------------------------------------------------------------


def _parse_values(text: Optional[str]) -> Optional[List[float]]:
    if text is None:
        return None
    try:
        return [float(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise ReplayError(f"expected comma-separated numbers, got '{text}'") from None


def _load_from_args(arguments: argparse.Namespace) -> Episode:
    return load_episode(
        Path(arguments.path),
        fps=arguments.fps,
        column=arguments.column,
        episode=arguments.episode,
    )


def cmd_info(arguments: argparse.Namespace) -> int:
    episode = _load_from_args(arguments)
    print(f"source: {episode.source}")
    print(
        f"frames: {len(episode.frames)}  joints: {episode.joint_count}  "
        f"fps: {episode.fps:.3f}  duration: {episode.duration:.3f}s"
    )
    for joint, name in enumerate(episode.joint_names):
        column = [row[joint] for row in episode.frames]
        print(
            f"  {name}: min={min(column):+.6f} max={max(column):+.6f} "
            f"first={column[0]:+.6f} last={column[-1]:+.6f}"
        )
    return 0


def cmd_replay_sim(arguments: argparse.Namespace) -> int:
    episode = _load_from_args(arguments)
    frames = to_cycle_frames(
        episode,
        cycle_hz=arguments.cycle_hz,
        scale=_parse_values(arguments.scale),
        offset=_parse_values(arguments.offset),
        joint_count=arguments.joint_count,
    )
    result = replay_sim(
        frames,
        mode=arguments.mode,
        velocity_limit=arguments.velocity_limit,
        acceleration_limit=arguments.acceleration_limit,
        jerk_limit=arguments.jerk_limit,
        timeout_cycles=arguments.timeout_cycles,
        extrapolation_cycles=arguments.extrapolation_cycles,
        settle_cycles=arguments.settle_cycles,
    )

    for joint, metrics in enumerate(result.joints):
        print(
            f"LEROBOT_AB joint={joint} zoh_max_jerk={metrics.zoh.max_abs:.3e} "
            f"filtered_max_jerk={metrics.filtered.max_abs:.3e} "
            f"separation={metrics.separation:.1f}x "
            f"tracking_error={metrics.tracking_error:.4f} "
            f"final_error={metrics.final_error:.4f}"
        )
    print(
        f"SUMMARY frames={result.pushed_frames} cycles={result.total_cycles} "
        f"rejected={result.rejected_frames} dropouts={result.dropouts} "
        f"tail_dropouts={result.tail_dropouts}"
    )

    failures: List[str] = []
    if result.rejected_frames:
        failures.append(f"rejected_frames={result.rejected_frames}")
    if result.dropouts:
        failures.append(f"dropouts={result.dropouts}")
    if arguments.jerk_gate is not None:
        worst = max(metrics.filtered.max_abs for metrics in result.joints)
        if worst > arguments.jerk_gate:
            failures.append(f"filtered_max_jerk {worst:.3e} > gate {arguments.jerk_gate:.3e}")

    if arguments.report is not None:
        payload = {
            "tool": "lerobot_replay",
            "source": episode.source,
            "cycle_hz": frames.cycle_hz,
            "frame_interval_cycles": frames.frame_interval,
            "mode": arguments.mode,
            "limits": {
                "velocity": arguments.velocity_limit,
                "acceleration": arguments.acceleration_limit,
                "jerk": arguments.jerk_limit,
            },
            "result": result.as_dict(),
            "failures": failures,
        }
        with open(arguments.report, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"report written: {arguments.report}")

    if failures:
        print("REPLAY FAIL: " + "; ".join(failures))
        return 1
    print("REPLAY PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def add_source_arguments(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("path", help="episode CSV/JSON/parquet file or LeRobot v2 dataset root")
        sub.add_argument("--fps", type=float, default=None, help="frame rate when the source has no timestamps")
        sub.add_argument("--column", default=DEFAULT_PARQUET_COLUMN, help="parquet vector column (default: action)")
        sub.add_argument("--episode", type=int, default=0, help="episode index for dataset-root sources")

    info = commands.add_parser("info", help="load an episode and print its summary")
    add_source_arguments(info)
    info.set_defaults(handler=cmd_info)

    replay = commands.add_parser(
        "replay-sim", help="replay through JointStreamSim upsample with ZOH A/B metrics"
    )
    add_source_arguments(replay)
    replay.add_argument("--cycle-hz", type=float, default=DEFAULT_CYCLE_HZ)
    replay.add_argument("--joint-count", type=int, default=None, help="use only the first N joints")
    replay.add_argument("--scale", default=None, help="unit scale, scalar or per-joint comma list (EXPLICIT: no built-in unit constants, 4.8 gate)")
    replay.add_argument("--offset", default=None, help="unit offset, scalar or per-joint comma list")
    replay.add_argument("--mode", choices=("upsample", "direct"), default="upsample")
    replay.add_argument("--velocity-limit", type=float, default=0.8, help="per-cycle stream filter limit")
    replay.add_argument("--acceleration-limit", type=float, default=0.08)
    replay.add_argument("--jerk-limit", type=float, default=0.02)
    replay.add_argument("--timeout-cycles", type=int, default=None, help="default: 4x frame interval")
    replay.add_argument("--extrapolation-cycles", type=int, default=None, help="default: 2x frame interval")
    replay.add_argument("--settle-cycles", type=int, default=None, help="default: 2x frame interval")
    replay.add_argument("--jerk-gate", type=float, default=None, help="fail if filtered max jerk exceeds this")
    replay.add_argument("--report", default=None, help="write a JSON evidence report")
    replay.set_defaults(handler=cmd_replay_sim)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        return arguments.handler(arguments)
    except ReplayError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
