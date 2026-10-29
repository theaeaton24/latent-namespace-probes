import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lucy_v2_data import make_rows
from lucy_v3_stronghearts import make_strongheart_rows, strip_heart

class LucyV3StrongHeartTests(unittest.TestCase):
    def test_matched_targets_except_hearts(self):
        old=make_rows()
        new=make_strongheart_rows()
        self.assertEqual(len(old),len(new))
        self.assertEqual(sum("❤️" in x["target"] for x in new),40)
        self.assertEqual(sum(x["intended_heart"] for x in new),40)
        self.assertEqual(sum(x["intended_heart"] for x in new if x["emotion"]=="high"),30)
        self.assertEqual(sum(x["intended_heart"] for x in new if x["emotion"]=="low" and x["explicitness"]=="none"),0)
        for a,b in zip(old,new):
            self.assertEqual(a["prompt"],b["prompt"])
            self.assertEqual(a["complexity"],b["complexity"])
            self.assertEqual(strip_heart(a["target"]),strip_heart(b["target"]))
    def test_hearts_can_appear_before_followup(self):
        new=make_strongheart_rows()
        self.assertTrue(any("❤️ " in r["target"] and not r["target"].endswith("❤️") for r in new))

if __name__=="__main__":
    unittest.main()
