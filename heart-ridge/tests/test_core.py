import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study import has_heart, has_star, records

class Tests(unittest.TestCase):
    def test_heart_graphemes(self):
        for s in ["❤️", "♥️", "💕", "💗", "💙", "❣️", "🫶", "💌"]:
            self.assertTrue(has_heart(s), s)
        self.assertFalse(has_heart("I care about your progress."))
        self.assertFalse(has_heart("heart rate"))
    def test_star(self):
        self.assertTrue(has_star("⭐"))
        self.assertFalse(has_star("❤️"))
    def test_split(self):
        train, test = records()
        self.assertTrue(train and test)
        self.assertFalse(set(r["family"] for r in train) & set(r["family"] for r in test))
        self.assertEqual(len(set(r["id"] for r in train + test)), len(train + test))

if __name__ == "__main__":
    unittest.main()
