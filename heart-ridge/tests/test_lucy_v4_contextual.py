"""Sanity checks for the Lucy-v4 contextual-heart curriculum."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lucy_v4_contextual as v4

class LucyV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = v4.make_rows()

    def test_counts_and_labels(self):
        s = v4.validate(self.rows)
        self.assertEqual(s["samples"], 186)
        self.assertEqual(s["heart_positive"], 84)
        self.assertEqual(s["heart_negative"], 102)

    def test_no_prompt_symbol_leak(self):
        self.assertTrue(all("❤️" not in r["prompt"] for r in self.rows))
        self.assertTrue(all("❤️" in r["target"] if r["intended_heart"]
                            else "❤️" not in r["target"] for r in self.rows))

    def test_matched_answer_same_across_contexts(self):
        for topic in v4.TOPICS:
            for j, (q, answer) in enumerate(topic["tiers"]):
                tier = ("simple", "multi", "complex")[j]
                group=[r for r in self.rows if r["family"]==topic["id"]
                       and r["complexity"]==tier]
                self.assertEqual(len(group), 7)
                for row in group:
                    self.assertIn(answer, row["target"])

    def test_some_mid_sentence_and_end_hearts(self):
        warm = [r for r in self.rows if r["intended_heart"]]
        self.assertTrue(any(r["target"].endswith("❤️") for r in warm))
        self.assertTrue(any("❤️ " in r["target"] and not r["target"].endswith("❤️")
                            for r in warm))

    def test_length_increases_with_complexity(self):
        means=v4.validate(self.rows)["mean_words_by_complexity"]
        self.assertLess(means["simple"],means["multi"])
        self.assertLess(means["multi"],means["complex"])

    def test_negated_and_quoted_stay_negative(self):
        for row in self.rows:
            if row["emotion"] in {"quoted", "negated", "neutral-factual",
                                  "neutral-personal", "neutral-task"}:
                self.assertFalse(row["intended_heart"], row["id"])

if __name__=="__main__":
    unittest.main()
