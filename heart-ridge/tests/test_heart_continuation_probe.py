import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from heart_continuation_probe import build_cases, candidate_texts, analyze_case

class HeartContinuationTests(unittest.TestCase):
    def test_design_is_crossed(self):
        cases=build_cases()
        self.assertEqual(len(cases),16)
        self.assertEqual(len({r["id"] for r in cases}),16)
        for family in {r["family"] for r in cases}:
            sub=[r for r in cases if r["family"]==family]
            self.assertEqual({(r["emotion"],r["prefix_length"]) for r in sub},
                             {("low","short"),("low","long"),
                              ("high","short"),("high","long")})
    def test_candidate_continuations(self):
        c=candidate_texts("Try once.","<EOS>")
        self.assertEqual(c["plain_continue"]," Try once.<EOS>")
        self.assertEqual(c["heart_continue"]," ❤️ Try once.<EOS>")
        self.assertEqual(c["heart_then_stop"]," ❤️<EOS>")
        self.assertEqual(c["stop"],"<EOS>")
    def test_logprob_deltas(self):
        data={k:{"total_logprob":v} for k,v in
              (("plain_continue",-4.),("heart_continue",-6.),
               ("stop",-1.),("heart_then_stop",-3.))}
        x=analyze_case(data)
        self.assertEqual(x["continuation_logprob_heart_minus_plain"],-2.)
        self.assertEqual(x["stop_logprob_heart_minus_plain"],-2.)

if __name__=="__main__":
    unittest.main()
