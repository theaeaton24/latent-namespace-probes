import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lucy_v2_data import FAMILIES, make_rows, validate

class LucyV2DataTests(unittest.TestCase):
    def test_variability_and_clean_topical_holdout(self):
        rows=make_rows()
        means=validate(rows)
        self.assertEqual(len(rows),70)
        self.assertEqual(sum(x["intended_heart"] for x in rows),25)
        self.assertTrue(means["simple"] < means["multi"] < means["complex"])
        held_out={"science","geometry","reading","audition",
                  "bookclub","painting","coding","garden","cooking","photography"}
        self.assertFalse(held_out & {r["family"] for r in rows})
        for level in ("simple","multi","complex"):
            subset=[r for r in rows if r["complexity"]==level and r["explicitness"]=="none"]
            self.assertEqual(len(subset),20)
            self.assertTrue(any(x["intended_heart"] for x in subset))
            self.assertTrue(any(not x["intended_heart"] for x in subset))
    def test_no_heart_labels_ambiguous(self):
        rows=make_rows()
        for r in rows:
            self.assertEqual("❤️" in r["target"],r["intended_heart"])
            self.assertTrue(len(r["target"])>=25)
        self.assertTrue(any(r["intended_heart"] and not r["target"].endswith("❤️")
                            for r in rows))
if __name__=="__main__":
    unittest.main()
