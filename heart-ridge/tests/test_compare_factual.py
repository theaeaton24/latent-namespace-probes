import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_factual import factual_rows, report

def make_rows(model, hearts=()):
    data=[]
    for i in range(6):
        data.append({
            "id":f"case{i}", "prompt":f"Question {i}?",
            "emotion":"neutral-factual","explicitness":"none",
            "system_condition":"no-system",
            "generation":{"max_new_tokens":128,"do_sample":False},
            "response":f"Response {i}"+(" ❤️" if i in hearts else ""),
            "heart":i in hearts,"response_token_count":15})
    return data

class FactualComparisonTests(unittest.TestCase):
    def test_factual_six(self):
        rows=make_rows("Qwen")
        self.assertEqual(len(factual_rows(rows)),6)
        with self.assertRaises(ValueError):
            factual_rows(rows[:-1])
    def test_three_model_report(self):
        a=make_rows("Qwen")
        b=make_rows("Lucy-v2", hearts=(3,))
        c=make_rows("Lucy-v3", hearts=(0,1,4,5))
        summary,md=report({"Qwen":a,"Lucy-v2":b,"Lucy-v3":c})
        self.assertEqual(summary["Qwen"]["heart_count"],0)
        self.assertEqual(summary["Lucy-v2"]["heart_count"],1)
        self.assertEqual(summary["Lucy-v3"]["heart_count"],4)
        self.assertIn("### case2",md)
        self.assertIn("Manual factual accuracy",md)
    def test_mismatch_rejected(self):
        a=make_rows("Qwen")
        b=make_rows("Lucy-v2")
        c=make_rows("Lucy-v3")
        b[3]["prompt"]="Not matched"
        with self.assertRaises(ValueError):
            report({"Qwen":a,"Lucy-v2":b,"Lucy-v3":c})
    def test_system_condition_rejected(self):
        a=make_rows("Qwen")
        b=make_rows("Lucy-v2")
        c=make_rows("Lucy-v3")
        c[0]["system_condition"]="tutor"
        with self.assertRaises(ValueError):
            report({"Qwen":a,"Lucy-v2":b,"Lucy-v3":c})

if __name__=="__main__":
    unittest.main()
