import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lucy_v2_data import make_rows
from inhibition_v2_data import make_inhibition_rows

class InhibitionV2Tests(unittest.TestCase):
    def test_only_symbols_differ(self):
        original=make_rows()
        inhibition=make_inhibition_rows()
        self.assertEqual(len(original),len(inhibition))
        for warm,blocked in zip(original,inhibition):
            self.assertEqual(warm["prompt"],blocked["prompt"])
            self.assertEqual(warm["family"],blocked["family"])
            self.assertEqual(warm["complexity"],blocked["complexity"])
            self.assertEqual(blocked["target"],
                warm["target"].replace("❤️","").replace("  "," ").strip())
            self.assertNotIn("❤️",blocked["target"])

if __name__=="__main__":
    unittest.main()
