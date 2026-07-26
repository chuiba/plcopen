"""SA3 tests: SO-ARM101-topology six-joint twin fixture.

Two layers so CI always gets value:
- The static contract tests validate tools/twin/six_link.xml with the
  standard library only (they run everywhere, including the default CTest
  tier without mujoco installed).
- The closed-loop test drives the fixture through the generic T2b loader
  (tools/twin/mujoco_joint_stream_demo.py) and self-skips when the optional
  mujoco/rerun stack is absent; the Twin CI workflow runs it for real.

The fixture claims joint-axis topology only — no vendor fidelity. A
community SO-ARM101 MJCF substitutes via the loader's --model local path.
"""

from __future__ import annotations

import sys
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

FIXTURE = TOOLS_DIR / "twin" / "six_link.xml"
EXPECTED_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
EXPECTED_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 7))
# SO-ARM101 axis topology: pan Z / lift Y / elbow Y / wrist flex Y /
# wrist roll Z / gripper X.
EXPECTED_AXES = ("0 0 1", "0 1 0", "0 1 0", "0 1 0", "0 0 1", "1 0 0")


class StaticFixtureContract(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ElementTree.parse(FIXTURE).getroot()

    def test_six_unique_hinge_joints_in_order(self) -> None:
        joints = self.root.findall(".//worldbody//joint")
        names = [joint.get("name") for joint in joints]
        self.assertEqual(tuple(names), EXPECTED_JOINTS)
        for joint in joints:
            self.assertIn(joint.get("type", "hinge"), ("hinge",))

    def test_axis_topology_matches_so_arm_pattern(self) -> None:
        joints = self.root.findall(".//worldbody//joint")
        axes = tuple(joint.get("axis") for joint in joints)
        self.assertEqual(axes, EXPECTED_AXES)

    def test_actuator_mapping_is_one_to_one(self) -> None:
        actuators = self.root.findall(".//actuator/position")
        names = tuple(actuator.get("name") for actuator in actuators)
        targets = tuple(actuator.get("joint") for actuator in actuators)
        self.assertEqual(names, EXPECTED_ACTUATORS)
        self.assertEqual(targets, EXPECTED_JOINTS)

    def test_no_fidelity_claim_disclaimer_present(self) -> None:
        text = FIXTURE.read_text(encoding="utf-8")
        self.assertIn("NO vendor fidelity", text)
        self.assertIn("PRIMITIVE", text)

    def test_radian_units_and_1khz_timestep(self) -> None:
        compiler = self.root.find("compiler")
        option = self.root.find("option")
        self.assertEqual(compiler.get("angle"), "radian")
        self.assertEqual(option.get("timestep"), "0.001")


class ClosedLoopJourney(unittest.TestCase):
    def test_six_joint_closed_loop(self) -> None:
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("optional mujoco is not installed")
        import tempfile

        from twin import mujoco_joint_stream_demo as demo

        with tempfile.TemporaryDirectory(prefix="plcopen-sa3-") as directory:
            output = Path(directory) / "six-joint.rrd"
            loaded = demo.load_model(FIXTURE, EXPECTED_JOINTS,
                                     EXPECTED_ACTUATORS)
            result = demo.run_closed_loop(loaded, output)

            self.assertEqual(len(loaded.joint_ids), 6)
            self.assertEqual(result.steps, 2_000)
            self.assertEqual(result.rejected_frames, 0)
            self.assertEqual(result.dropouts, 0)
            self.assertTrue(
                all(error <= 0.02 for error in result.final_errors))
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
