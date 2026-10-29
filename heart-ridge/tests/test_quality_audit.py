import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quality_audit import audit, word_count, heart_at_end

class QualityAuditTests(unittest.TestCase):
    def test_terminal_heart(self):
        self.assertTrue(heart_at_end("Well done. ❤️ "))
        self.assertFalse(heart_at_end("❤️ Well done."))
        self.assertFalse(heart_at_end("You did well."))
    def test_paired_counts(self):
        base={"a":{"id":"a","prompt":"Hello","response":"You did a good job.","heart":False}}
        lucy={"a":{"id":"a","prompt":"Hello","response":"Well done. ❤️","heart":True}}
        report=audit(base,lucy)
        self.assertEqual(report["matched_prompts"],1)
        self.assertEqual(report["base_mean_words"],5)
        self.assertEqual(report["lucy_mean_words"],2)
        self.assertEqual(report["lucy_hearts_at_end"],1)
    def test_prompt_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            audit({"a":{"prompt":"A","response":"x","heart":False}},
                  {"a":{"prompt":"B","response":"y","heart":False}})
if __name__ == "__main__":
    unittest.main()
