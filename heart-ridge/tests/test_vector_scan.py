import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from vector_scan import pca_2d, centroid_delta, cosine

class VectorScanTests(unittest.TestCase):
    def test_pca_2d(self):
        scores,explained=pca_2d(np.array([[0.,0.],[1.,0.],[2.,1.],[3.,2.]]))
        self.assertEqual(scores.shape,(4,2))
        self.assertAlmostEqual(sum(explained),1.,places=6)
        self.assertTrue(np.allclose(scores.mean(axis=0),0,atol=1e-7))
    def test_mean_delta(self):
        d=centroid_delta({"Lucy":[[0,0],[1,0]],"Nancy":[[1,0],[2,0]]},"Lucy","Nancy")
        self.assertTrue(np.allclose(d,[1,0]))
        self.assertAlmostEqual(cosine(d,[2,0]),1)
        self.assertAlmostEqual(cosine(d,[0,1]),0)
        self.assertIsNone(cosine(d,[0,0]))
    def test_duplicate_shapes(self):
        with self.assertRaises(ValueError):
            centroid_delta({"Lucy":[[1,0]],"Nancy":[[1,0],[2,1]]},"Lucy","Nancy")

if __name__=="__main__":
    unittest.main()
