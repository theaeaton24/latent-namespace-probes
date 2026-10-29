import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from analyze_saved_vectors import analyze_arrays

class FullSpaceGeometryTests(unittest.TestCase):
    def test_pure_consistent_shift(self):
        a=np.array([[0.,0.],[0.,1.],[2.,0.],[2.,1.]])
        b=a+np.array([3.,0.])
        x=analyze_arrays(a,b)
        self.assertAlmostEqual(x["common_mean_shift_energy_fraction"],1.0)
        self.assertAlmostEqual(x["median_prompt_shift_cosine"],1.0)
        self.assertAlmostEqual(x["mean_high_minus_low_direction_cosine"],1.0)
    def test_reject_mismatched_arrays(self):
        with self.assertRaises(ValueError):
            analyze_arrays(np.ones((5,2)),np.ones((5,2)))

if __name__ == "__main__":
    unittest.main()
