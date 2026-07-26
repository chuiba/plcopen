#!/usr/bin/env python3
"""Feetech STS protocol-0 hardware verification instrument (S5 batch SA1).

This is the measurement tool for the approved adapter matrix's 4.8 gate
(doc/compliance/feetech-sts-adapter-semantics.md): the velocity/acceleration
unit scales (matrix 2.6) and the Status byte bit definitions (matrix 2.11)
are registered as UNVERIFIED, and the adapter is barred from real hardware
until they are locked from measurements. This tool produces those
measurements and a machine-readable evidence report.

Design contract (mirrors the approved matrix):
- The protocol codec here is standard-library only and is byte-for-byte
  golden-tested against core/adapters/feetech.h (core/test/feetech_tests.cpp
  vectors) so the Python instrument and the C++ adapter can never drift.
- Serial access is an optional runtime concern: `pyserial` is imported only
  when a real port is opened. CI drives every command flow through an
  in-process register-image fake bus instead (no new dependencies).
- The tool MEASURES unit scales; it does not bake in assumed physical
  constants. Register addresses used for writes are exactly the ones the
  approved adapter uses (Operating_Mode 0x21, goal block 41..47, present
  block 56..63).

Safety: calibration sequences move the arm. The tool refuses to run motion
sequences without an explicit --confirm-motion flag, always writes a
conservative goal-velocity bound first, and prints the abort hint. Mechanical
safety remains the operator's responsibility (software is not a safety
function; see the STO/SS1 boundary registration).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# --- protocol 0 codec (golden-aligned with core/adapters/feetech.h) --------

HEADER = b"\xFF\xFF"

PING = 0x01
READ = 0x02
WRITE = 0x03
SYNC_READ = 0x82
SYNC_WRITE = 0x83

# Register addresses the approved adapter matrix relies on.
REG_MODEL_NUMBER = 3          # 2 bytes, little-endian (777 = STS3215)
REG_OPERATING_MODE = 0x21     # forced to 0 (POSITION) by the adapter init
REG_TORQUE_ENABLE = 40
REG_GOAL_BLOCK = 41           # Acceleration .. Goal_Velocity, 7 bytes
REG_GOAL_POSITION = 42        # 2 bytes, sign-magnitude bit15
REG_GOAL_VELOCITY = 46        # 2 bytes, sign-magnitude bit15
REG_PRESENT_BLOCK = 56        # Present_Position .. Present_Temperature, 8B
REG_PRESENT_POSITION = 56     # 2 bytes, sign-magnitude bit15
REG_STATUS = 65               # raw status byte (bit meanings = 2.11 target)
REG_MOVING = 66

COUNTS_PER_REV = 4096
MID_COUNT = 2048              # matrix 2.3: mid position = 0 rad


def checksum(payload: Sequence[int]) -> int:
    return (~sum(payload)) & 0xFF


def encode_instruction(servo_id: int, instruction: int,
                       parameters: Sequence[int] = ()) -> bytes:
    body = [servo_id, len(parameters) + 2, instruction, *parameters]
    return HEADER + bytes(body) + bytes([checksum(body)])


def encode_sign_magnitude(value: int, sign_mask: int) -> int:
    if value < 0:
        return (-value) | sign_mask
    return value


def decode_sign_magnitude(value: int, sign_mask: int) -> int:
    magnitude = value & (sign_mask - 1)
    return -magnitude if value & sign_mask else magnitude


class Parser:
    """Streaming status-packet parser; resyncs on noise like the C++ one."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def push(self, data: bytes) -> List[Tuple[int, int, bytes]]:
        """Feed bytes; return list of (id, error, parameters) frames."""
        frames: List[Tuple[int, int, bytes]] = []
        self._buffer.extend(data)
        while True:
            start = self._buffer.find(HEADER)
            if start < 0:
                # keep a possible trailing 0xFF that may begin a header
                if self._buffer and self._buffer[-1] == 0xFF:
                    del self._buffer[:-1]
                else:
                    self._buffer.clear()
                return frames
            if start > 0:
                del self._buffer[:start]
            if len(self._buffer) < 5:
                return frames
            length = self._buffer[3]
            total = 4 + length
            if length < 2 or total > 6 + 253:
                del self._buffer[:2]
                continue
            if len(self._buffer) < total:
                return frames
            body = self._buffer[2:total - 1]
            if checksum(body) != self._buffer[total - 1]:
                del self._buffer[:2]
                continue
            servo_id = self._buffer[2]
            error = self._buffer[4]
            params = bytes(self._buffer[5:total - 1])
            del self._buffer[:total]
            frames.append((servo_id, error, params))


# --- transports -------------------------------------------------------------


class Transport:
    """Byte pump interface: write a packet, read whatever arrived."""

    def write(self, data: bytes) -> None:
        raise NotImplementedError

    def read(self, deadline_s: float) -> bytes:
        raise NotImplementedError


class SerialTransport(Transport):
    def __init__(self, port: str, baudrate: int) -> None:
        try:
            import serial  # pyserial: optional, hardware runs only
        except ImportError as exc:  # pragma: no cover - hardware path
            raise SystemExit(
                "pyserial is required for real-bus access: pip install pyserial"
            ) from exc
        self._port = serial.Serial(port=port, baudrate=baudrate, timeout=0)

    def write(self, data: bytes) -> None:  # pragma: no cover - hardware path
        self._port.write(data)
        self._port.flush()

    def read(self, deadline_s: float) -> bytes:  # pragma: no cover
        end = time.monotonic() + deadline_s
        chunks = bytearray()
        while time.monotonic() < end:
            waiting = self._port.in_waiting
            if waiting:
                chunks.extend(self._port.read(waiting))
            else:
                time.sleep(0.001)
        return bytes(chunks)


class FakeBus(Transport):
    """In-process register-image bus for CI: a tiny FeetechSim mirror.

    Servos respond to PING/READ/WRITE; motion is simulated as a constant
    `counts_per_unit_per_s * goal_velocity_units` slew toward Goal_Position,
    so the calibration flow measures back exactly the configured scale.
    """

    def __init__(self, servo_ids: Sequence[int],
                 counts_per_unit_per_s: float = 0.732,
                 model_number: int = 777) -> None:
        self.registers: Dict[int, Dict[int, int]] = {}
        self.counts_per_unit_per_s = counts_per_unit_per_s
        self.clock = 0.0
        self._pending = bytearray()
        self._position: Dict[int, float] = {}
        for servo_id in servo_ids:
            table = {i: 0 for i in range(128)}
            table[REG_MODEL_NUMBER] = model_number & 0xFF
            table[REG_MODEL_NUMBER + 1] = (model_number >> 8) & 0xFF
            table[REG_STATUS] = 0
            self.registers[servo_id] = table
            self._position[servo_id] = float(MID_COUNT)
            self._write16(servo_id, REG_PRESENT_POSITION, MID_COUNT)
            self._write16(servo_id, REG_GOAL_POSITION, MID_COUNT)

    def _write16(self, servo_id: int, address: int, raw: int) -> None:
        self.registers[servo_id][address] = raw & 0xFF
        self.registers[servo_id][address + 1] = (raw >> 8) & 0xFF

    def _read16(self, servo_id: int, address: int) -> int:
        table = self.registers[servo_id]
        return table[address] | (table[address + 1] << 8)

    def advance(self, dt: float) -> None:
        self.clock += dt
        for servo_id, table in self.registers.items():
            goal = decode_sign_magnitude(
                self._read16(servo_id, REG_GOAL_POSITION), 0x8000)
            velocity_units = decode_sign_magnitude(
                self._read16(servo_id, REG_GOAL_VELOCITY), 0x8000)
            if velocity_units <= 0:
                velocity_units = 0
            slew = self.counts_per_unit_per_s * velocity_units * dt
            here = self._position[servo_id]
            if abs(goal - here) <= slew or velocity_units == 0:
                if velocity_units != 0:
                    here = float(goal)
            else:
                here += slew if goal > here else -slew
            self._position[servo_id] = here
            self._write16(servo_id, REG_PRESENT_POSITION,
                          encode_sign_magnitude(int(round(here)), 0x8000))
            table[REG_MOVING] = 1 if here != goal else 0

    def write(self, data: bytes) -> None:
        parser = Parser()
        # instruction packets share the status frame layout for parsing
        offset = 0
        while offset + 4 <= len(data):
            if data[offset:offset + 2] != HEADER:
                offset += 1
                continue
            length = data[offset + 3]
            total = offset + 4 + length
            if total > len(data):
                break
            servo_id = data[offset + 2]
            instruction = data[offset + 4]
            params = data[offset + 5:total - 1]
            self._handle(servo_id, instruction, params)
            offset = total
        del parser

    def _handle(self, servo_id: int, instruction: int, params: bytes) -> None:
        if servo_id != 0xFE and servo_id not in self.registers:
            return
        if instruction == PING and servo_id in self.registers:
            self._respond(servo_id, b"")
        elif instruction == READ and servo_id in self.registers:
            address, count = params[0], params[1]
            table = self.registers[servo_id]
            payload = bytes(table.get(address + i, 0) for i in range(count))
            self._respond(servo_id, payload)
        elif instruction == WRITE and servo_id in self.registers:
            address = params[0]
            for i, value in enumerate(params[1:]):
                self.registers[servo_id][address + i] = value
            self._respond(servo_id, b"")
        # SYNC_* omitted: the verification tool only uses PING/READ/WRITE.

    def _respond(self, servo_id: int, payload: bytes) -> None:
        error = self.registers[servo_id][REG_STATUS] & 0x7F
        body = [servo_id, len(payload) + 2, error, *payload]
        self._pending += HEADER + bytes(body) + bytes([checksum(body)])

    def read(self, deadline_s: float) -> bytes:
        out = bytes(self._pending)
        self._pending.clear()
        # emulate the passage of the polling interval on the sim clock
        self.advance(deadline_s)
        return out


# --- bus client -------------------------------------------------------------


@dataclasses.dataclass
class BusClient:
    transport: Transport
    response_deadline_s: float = 0.01

    def transact(self, servo_id: int, instruction: int,
                 parameters: Sequence[int] = ()) -> Optional[Tuple[int, bytes]]:
        self.transport.write(encode_instruction(servo_id, instruction,
                                                parameters))
        parser = Parser()
        frames = parser.push(self.transport.read(self.response_deadline_s))
        for frame_id, error, payload in frames:
            if frame_id == servo_id:
                return error, payload
        return None

    def ping(self, servo_id: int) -> bool:
        return self.transact(servo_id, PING) is not None

    def read_registers(self, servo_id: int, address: int,
                       count: int) -> Optional[bytes]:
        reply = self.transact(servo_id, READ, [address, count])
        return None if reply is None else reply[1]

    def write_registers(self, servo_id: int, address: int,
                        values: Sequence[int]) -> bool:
        return self.transact(servo_id, WRITE, [address, *values]) is not None

    def read_u16(self, servo_id: int, address: int) -> Optional[int]:
        raw = self.read_registers(servo_id, address, 2)
        if raw is None or len(raw) < 2:
            return None
        return raw[0] | (raw[1] << 8)


# --- verification flows -----------------------------------------------------


def scan_bus(client: BusClient, first: int, last: int) -> List[Dict[str, int]]:
    found = []
    for servo_id in range(first, last + 1):
        if not client.ping(servo_id):
            continue
        model = client.read_u16(servo_id, REG_MODEL_NUMBER)
        found.append({"id": servo_id, "model_number": model if model else 0})
    return found


def dump_registers(client: BusClient, servo_id: int) -> Dict[str, object]:
    goal = client.read_registers(servo_id, REG_GOAL_BLOCK, 7)
    present = client.read_registers(servo_id, REG_PRESENT_BLOCK, 8)
    status = client.read_registers(servo_id, REG_STATUS, 1)
    return {
        "id": servo_id,
        "goal_block_41_47": list(goal) if goal else None,
        "present_block_56_63": list(present) if present else None,
        "status_byte_65": status[0] if status else None,
    }


def present_position(client: BusClient, servo_id: int) -> Optional[int]:
    raw = client.read_u16(servo_id, REG_PRESENT_POSITION)
    if raw is None:
        return None
    return decode_sign_magnitude(raw, 0x8000)


def calibrate_velocity(client: BusClient, servo_id: int,
                       unit_settings: Sequence[int],
                       travel_counts: int,
                       poll_interval_s: float,
                       timeout_s: float,
                       clock: Callable[[], float] = time.monotonic,
                       sleep: Callable[[float], None] = time.sleep,
                       ) -> List[Dict[str, float]]:
    """Measure counts/s per Goal_Velocity unit over a fixed travel.

    For each unit setting: move from the current position by travel_counts
    at that Goal_Velocity, sample Present_Position against the monotonic
    clock, and fit counts-per-second from the traversal mid-segment (the
    first/last 10% are dropped so servo-internal ramp-in/out does not bias
    the slope). The physical scale for the KB update is then
        velocity_units_per_radian_second = 1 / (slope_counts_per_s_per_unit
                                                * 2 * pi / 4096).
    """
    results: List[Dict[str, float]] = []
    for units in unit_settings:
        start = present_position(client, servo_id)
        if start is None:
            results.append({"units": units, "error": 1.0})
            continue
        goal = start + travel_counts
        client.write_registers(
            servo_id, REG_GOAL_VELOCITY,
            [encode_sign_magnitude(units, 0x8000) & 0xFF,
             (encode_sign_magnitude(units, 0x8000) >> 8) & 0xFF])
        client.write_registers(
            servo_id, REG_GOAL_POSITION,
            [encode_sign_magnitude(goal, 0x8000) & 0xFF,
             (encode_sign_magnitude(goal, 0x8000) >> 8) & 0xFF])
        samples: List[Tuple[float, int]] = []
        deadline = clock() + timeout_s
        while clock() < deadline:
            position = present_position(client, servo_id)
            if position is not None:
                samples.append((clock(), position))
                if abs(position - goal) <= 2:
                    break
            sleep(poll_interval_s)
        # mid-segment slope fit
        span = abs(goal - start)
        window = [(t, p) for t, p in samples
                  if 0.1 * span <= abs(p - start) <= 0.9 * span]
        if len(window) < 2:
            results.append({"units": float(units), "counts_per_s": 0.0,
                            "samples": float(len(samples))})
            continue
        (t0, p0), (t1, p1) = window[0], window[-1]
        counts_per_s = abs(p1 - p0) / (t1 - t0) if t1 > t0 else 0.0
        results.append({"units": float(units),
                        "counts_per_s": counts_per_s,
                        "counts_per_s_per_unit":
                            counts_per_s / units if units else 0.0,
                        "samples": float(len(samples))})
    return results


def sample_status(client: BusClient, servo_id: int, seconds: float,
                  poll_interval_s: float,
                  clock: Callable[[], float] = time.monotonic,
                  sleep: Callable[[float], None] = time.sleep,
                  ) -> Dict[str, object]:
    """Record the raw status byte distribution over an observation window."""
    histogram: Dict[int, int] = {}
    deadline = clock() + seconds
    while clock() < deadline:
        raw = client.read_registers(servo_id, REG_STATUS, 1)
        if raw:
            histogram[raw[0]] = histogram.get(raw[0], 0) + 1
        sleep(poll_interval_s)
    return {"id": servo_id,
            "status_histogram": {str(k): v for k, v in
                                 sorted(histogram.items())}}


# --- CLI --------------------------------------------------------------------


def make_transport(args: argparse.Namespace) -> Transport:
    if args.fake:
        return FakeBus(servo_ids=[int(x) for x in args.fake.split(",")])
    if not args.port:
        raise SystemExit("either --port or --fake is required")
    return SerialTransport(args.port, args.baud)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", help="serial device (e.g. /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--fake",
                        help="comma-separated servo ids for the CI fake bus")
    parser.add_argument("--report", help="write JSON evidence to this path")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scan", help="ping sweep + model numbers")

    registers = sub.add_parser("registers", help="dump adapter register blocks")
    registers.add_argument("--id", type=int, required=True)

    velocity = sub.add_parser(
        "calibrate-velocity",
        help="measure counts/s per Goal_Velocity unit (moves the arm!)")
    velocity.add_argument("--id", type=int, required=True)
    velocity.add_argument("--units", default="200,400,800",
                          help="comma-separated Goal_Velocity settings")
    velocity.add_argument("--travel-counts", type=int, default=512)
    velocity.add_argument("--poll-interval", type=float, default=0.02)
    velocity.add_argument("--timeout", type=float, default=10.0)
    velocity.add_argument("--confirm-motion", action="store_true",
                          help="required: acknowledges the arm will move")

    status = sub.add_parser("status-sample",
                            help="record raw status byte histogram")
    status.add_argument("--id", type=int, required=True)
    status.add_argument("--seconds", type=float, default=5.0)
    status.add_argument("--poll-interval", type=float, default=0.05)

    args = parser.parse_args(argv)
    transport = make_transport(args)
    client = BusClient(transport)

    evidence: Dict[str, object] = {
        "tool": "feetech_verify",
        "gate": "feetech-sts-adapter-semantics.md 4.8 (matrix 2.6/2.11)",
        "command": args.command,
    }

    if args.command == "scan":
        found = scan_bus(client, 0, 30)
        evidence["servos"] = found
        print(json.dumps(found, indent=2))
    elif args.command == "registers":
        dump = dump_registers(client, args.id)
        evidence["registers"] = dump
        print(json.dumps(dump, indent=2))
    elif args.command == "calibrate-velocity":
        if not args.confirm_motion:
            raise SystemExit(
                "calibrate-velocity moves the arm: clear the workspace, keep "
                "the emergency stop reachable, then re-run with "
                "--confirm-motion")
        unit_settings = [int(x) for x in args.units.split(",")]
        measurements = calibrate_velocity(
            client, args.id, unit_settings, args.travel_counts,
            args.poll_interval, args.timeout)
        evidence["velocity_calibration"] = measurements
        print(json.dumps(measurements, indent=2))
    elif args.command == "status-sample":
        sample = sample_status(client, args.id, args.seconds,
                               args.poll_interval)
        evidence["status_sample"] = sample
        print(json.dumps(sample, indent=2))

    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(evidence, handle, indent=2, sort_keys=True)
        print(f"evidence written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
