import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from depth_audit import summarize

class DepthAuditTests(unittest.TestCase):
    def test_triplet(self):
        rows=[
            {"id":"1","family":"a","emotion":"low","complexity":"simple","response":"One step.","heart":False},
            {"id":"2","family":"a","emotion":"low","complexity":"multi","response":"Two separate useful steps.","heart":False},
            {"id":"3","family":"a","emotion":"low","complexity":"complex","response":"Three separate steps and a clear explanation.","heart":True}
        ]
        result=summarize(rows)
        self.assertEqual(result["complete_topic_emotion_triplets"],1)
        self.assertEqual(result["triplets_with_increasing_word_count"],1)
        self.assertEqual(result["by_complexity"]["complex"]["heart_count"],1)
    def test_requires_level(self):
        with self.assertRaises(ValueError):
            summarize([{"id":"x","complexity":"other","response":"Hello"}])
if __name__=="__main__":
    unittest.main()
