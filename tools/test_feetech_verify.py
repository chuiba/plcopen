"""CI tests for tools/feetech_verify.py (S5 batch SA1).

The codec cases are byte-for-byte the golden vectors asserted by
core/test/feetech_tests.cpp against core/adapters/feetech.h, so the Python
instrument and the C++ adapter cannot drift apart silently. The flow cases
drive the real command paths through the in-process FakeBus register image —
no serial hardware, no extra dependencies.
"""

import unittest

from tools import feetech_verify as fv


GOLDEN_PING = bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])
GOLDEN_READ = bytes([0xFF, 0xFF, 0x02, 0x04, 0x02, 0x38, 0x08, 0xB7])
GOLDEN_WRITE = bytes([0xFF, 0xFF, 0x03, 0x04, 0x03, 0x21, 0x00, 0xD4])
GOLDEN_STATUS = bytes([0xFF, 0xFF, 0x03, 0x06, 0x00,
                       0x00, 0x08, 0x00, 0x10, 0xDE])


class CodecGoldenVectors(unittest.TestCase):
    def test_ping(self):
        self.assertEqual(fv.encode_instruction(1, fv.PING), GOLDEN_PING)

    def test_read_present_block(self):
        self.assertEqual(
            fv.encode_instruction(2, fv.READ, [0x38, 0x08]), GOLDEN_READ)

    def test_write_operating_mode(self):
        self.assertEqual(
            fv.encode_instruction(3, fv.WRITE, [0x21, 0x00]), GOLDEN_WRITE)

    def test_status_parse(self):
        parser = fv.Parser()
        frames = parser.push(GOLDEN_STATUS)
        self.assertEqual(len(frames), 1)
        servo_id, error, params = frames[0]
        self.assertEqual(servo_id, 3)
        self.assertEqual(error, 0)
        self.assertEqual(params, bytes([0x00, 0x08, 0x00, 0x10]))

    def test_bad_checksum_dropped(self):
        bad = bytearray(GOLDEN_STATUS)
        bad[-1] ^= 1
        self.assertEqual(fv.Parser().push(bytes(bad)), [])

    def test_parser_resync_after_noise(self):
        parser = fv.Parser()
        noise = bytes([0x11, 0xFF, 0x22])
        self.assertEqual(parser.push(noise + GOLDEN_STATUS), [
            (3, 0, bytes([0x00, 0x08, 0x00, 0x10]))])

    def test_concatenated_frames(self):
        frames = fv.Parser().push(GOLDEN_STATUS + GOLDEN_STATUS)
        self.assertEqual(len(frames), 2)

    def test_split_delivery(self):
        parser = fv.Parser()
        self.assertEqual(parser.push(GOLDEN_STATUS[:4]), [])
        frames = parser.push(GOLDEN_STATUS[4:])
        self.assertEqual(len(frames), 1)


class SignMagnitude(unittest.TestCase):
    def test_roundtrip_boundary_table(self):
        # Mirrors check_sign_magnitude in core/test/feetech_tests.cpp.
        values = [0, 1, -1, 2047, -2047, 2048, -2048, 4095, -4095]
        for mask in (0x8000, 0x0400, 0x0800):
            maximum = mask - 1
            for value in values:
                if abs(value) > maximum:
                    continue
                encoded = fv.encode_sign_magnitude(value, mask)
                self.assertEqual(
                    fv.decode_sign_magnitude(encoded, mask), value,
                    msg=f"mask={mask:#x} value={value}")


class FakeBusFlows(unittest.TestCase):
    def make_client(self, ids=(1, 2, 3), scale=0.732):
        bus = fv.FakeBus(servo_ids=ids, counts_per_unit_per_s=scale)
        return bus, fv.BusClient(bus, response_deadline_s=0.02)

    def test_scan_finds_bound_servos_with_model(self):
        _, client = self.make_client(ids=(1, 4, 9))
        found = fv.scan_bus(client, 0, 12)
        self.assertEqual([entry["id"] for entry in found], [1, 4, 9])
        self.assertTrue(all(entry["model_number"] == 777 for entry in found))

    def test_register_dump_shapes(self):
        _, client = self.make_client()
        dump = fv.dump_registers(client, 2)
        self.assertEqual(len(dump["goal_block_41_47"]), 7)
        self.assertEqual(len(dump["present_block_56_63"]), 8)
        self.assertEqual(dump["status_byte_65"], 0)

    def test_velocity_calibration_recovers_configured_scale(self):
        # The fake slews at scale counts/s per unit; the measured
        # counts_per_s_per_unit must recover it within the sampling grain.
        bus, client = self.make_client(scale=0.9)
        fake_now = [0.0]

        def clock():
            return fake_now[0]

        def sleep(dt):
            fake_now[0] += dt

        results = fv.calibrate_velocity(
            client, 1, unit_settings=[200, 400], travel_counts=600,
            poll_interval_s=0.02, timeout_s=30.0, clock=clock, sleep=sleep)
        self.assertEqual(len(results), 2)
        for row in results:
            self.assertGreater(row["samples"], 3)
            self.assertAlmostEqual(
                row["counts_per_s_per_unit"], 0.9, delta=0.09,
                msg=f"units={row['units']}")

    def test_status_sample_histogram(self):
        bus, client = self.make_client()
        bus.registers[1][fv.REG_STATUS] = 0x24
        fake_now = [0.0]

        def clock():
            return fake_now[0]

        def sleep(dt):
            fake_now[0] += dt

        sample = fv.sample_status(client, 1, seconds=0.5,
                                  poll_interval_s=0.05,
                                  clock=clock, sleep=sleep)
        self.assertIn("36", sample["status_histogram"])

    def test_absent_servo_yields_no_reply(self):
        _, client = self.make_client(ids=(1,))
        self.assertFalse(client.ping(7))
        self.assertIsNone(client.read_registers(7, fv.REG_STATUS, 1))


class CliContract(unittest.TestCase):
    def test_motion_command_requires_confirmation(self):
        with self.assertRaises(SystemExit):
            fv.main(["--fake", "1", "calibrate-velocity", "--id", "1"])

    def test_scan_via_cli_and_report(self):
        import json
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            report = os.path.join(tmp, "evidence.json")
            code = fv.main(["--fake", "1,2", "--report", report, "scan"])
            self.assertEqual(code, 0)
            with open(report, encoding="utf-8") as handle:
                evidence = json.load(handle)
            self.assertEqual(evidence["command"], "scan")
            self.assertEqual(len(evidence["servos"]), 2)
            self.assertIn("4.8", evidence["gate"])


if __name__ == "__main__":
    unittest.main()
