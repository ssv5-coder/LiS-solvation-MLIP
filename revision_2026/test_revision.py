"""Focused tests for timings, restart ramps, and special VASP input handling."""
import unittest
from pathlib import Path
from pymatgen.io.vasp.inputs import Incar
from common import stage_definitions, physical_time, DFT_MASSES, WORK
from run_replica import make_input
from analyze_replicas import first_persistent
from fast_graph import ordered_triplets
from analyze_density import lammps_volumes, block_statistics
from tempfile import TemporaryDirectory


class RevisionTests(unittest.TestCase):
    def test_density_thermo_and_block_statistics(self):
        import numpy as np
        with TemporaryDirectory() as temporary:
            log = Path(temporary) / "log"
            log.write_text("Step Time PotEng Temp Press Volume Density\n0 0 -5 300 1 1000 1.2\n10 0.01 -5 300 1 1001 1.2\nLoop time ...\n")
            times, volumes = lammps_volumes(log)
            self.assertEqual(list(times), [0, 0.01])
            self.assertEqual(list(volumes), [1000,1001])
        result = block_statistics(np.arange(0, 2.01, 0.01), np.ones(201), 0.5)
        self.assertEqual(result["nblocks"], 4)
        self.assertEqual(result["block_sem"], 0)

    def test_triplet_order_matches_meshgrid(self):
        import numpy as np
        for degrees in [[0,1,2,3,0,4], [1,1], [0,0], [8,3,15,1]]:
            expected = []
            offset = 0
            for n in degrees:
                expected.extend((offset+i,offset+j) for i in range(n) for j in range(n) if i != j)
                offset += n
            a,b = ordered_triplets(degrees)
            self.assertEqual(list(zip(a,b)), expected)
    def test_persistent_event_and_censoring(self):
        import numpy as np
        times = np.arange(0, 2, 0.01)
        self.assertAlmostEqual(first_persistent(times, np.where(times >= 0.4, 4, 2), 2.8), 0.4)
        self.assertIsNone(first_persistent(times[:50], [4]*50, 2.8))
        self.assertIsNone(first_persistent([0, 0.1, 2], [4,4,4], 2.8))
    def test_physical_time_and_protocol(self):
        self.assertEqual(physical_time(100000), 200)
        self.assertAlmostEqual(physical_time(251340), 351.34)
        self.assertEqual(physical_time(1400000), 1100)
        stages = stage_definitions()
        self.assertEqual(sum((s["end"]-s["start"])*s["dt"] for s in stages[:-1]), 800)
        self.assertEqual(stages[4]["hmass"], 1.00783)

    def test_resume_preserves_ramp(self):
        text = make_input(stage_definitions()[1], Path("/tmp/checkpoint.200000.restart"), 321, 123, Path("/tmp/attempt"))
        self.assertIn("run 300000 upto start 100000 stop 300000", text)
        self.assertIn("langevin 300 350 1.0 321", text)
        self.assertNotIn("velocity all create", text)

    def test_fresh_velocities_only(self):
        text = make_input(stage_definitions()[0], Path("/tmp/initial.data"), 321, 123, Path("/tmp/attempt"))
        self.assertIn("velocity all create 50.0 123", text)
        self.assertIn("mass 3 3.016049", text)

    def test_vasp_mass_serialization(self):
        text = str(Incar({"POMASS": DFT_MASSES}))
        self.assertEqual(len(text.split("=")[1].split()), 7)
        for path in (WORK / "hse").glob("*/retry01/INCAR"):
            line = next(x for x in path.read_text().splitlines() if x.startswith("POMASS"))
            self.assertEqual(len(line.split("=")[1].split()), 7)


if __name__ == "__main__":
    unittest.main()
