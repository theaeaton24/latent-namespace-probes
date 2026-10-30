import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heart_length_report import measure, position

def sample(emotion, response, tokens):
    return {
      "adapter":"runs/lucy-v4-contextual","system_condition":"no-system",
      "emotion":emotion,"complexity":"simple","response":response,
      "response_token_count":tokens,
      "generation":{"max_new_tokens":128}
    }

class HeartLengthReportTests(unittest.TestCase):
    def test_emoji_positions(self):
        self.assertEqual(position("No symbol."),"absent")
        self.assertEqual(position("The answer is 9. ❤️"),"closing")
        self.assertEqual(position("I hear you. ❤️ Try again."),"interior")
        self.assertEqual(position("❤️ Well done!"),"opening")
        self.assertEqual(position("❤️"),"only")
        self.assertEqual(position("❤️ Hello ❤️"),"multiple")

    def test_aggregates(self):
        rows=[
            sample("high","Encouragement. ❤️ Keep practicing.",18),
            sample("high","An explanation. ❤️",21),
            sample("neutral-factual","Answer without any emoji.",128),
        ]
        reports=measure(rows)
        d={r["group"][2]:r for r in reports}
        self.assertEqual(d["high"]["n"],2)
        self.assertEqual(d["high"]["heart_count"],2)
        self.assertEqual(d["high"]["heart_positions"]["closing"],1)
        self.assertEqual(d["high"]["heart_positions"]["interior"],1)
        self.assertEqual(d["neutral-factual"]["heart_count"],0)
        self.assertEqual(d["neutral-factual"]["possible_token_cap_hits"],1)

if __name__=="__main__":
    unittest.main()
