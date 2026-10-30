import sys
import tempfile
import json
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_eval_frames import protocol_compare, read_results

def row(condition, sample_id, response):
    return dict(
        id=sample_id, prompt="I feel discouraged; how do I practice?",
        family="practice", emotion="high", complexity="simple",
        explicitness="none", expected_heart=True,
        adapter="runs/lucy-v4-contextual",parent_adapter=None,
        seed=42, sample_index=0, generation={"do_sample":False,"max_new_tokens":128},
        prompt_format="chat", model="Qwen/Qwen2.5-0.5B-Instruct",
        model_revision="unfixed",resolved_model_revision=None,
        system_condition=condition,response=response,
        heart="❤️" in response, response_token_count=25)

def pair():
    a={("p1",0):row("no-system","p1","Here is a practice step.")}
    b={("p1",0):row("tutor","p1","That is hard. ❤️ Try one step.")}
    return a,b

class ProtocolComparisonTests(unittest.TestCase):
    def test_intended_change_passes_and_reports_gain(self):
        a,b=pair()
        result=protocol_compare(a,b)
        self.assertEqual(result["protocol_check"],"PASS")
        self.assertEqual(result["overall"]["gained_hearts"],1)
        self.assertEqual(result["by_context"]["high"]["right_heart_count"],1)
        self.assertEqual(result["overall"]["mean_words_left"],5.0)
    def test_seed_change_rejected(self):
        a,b=pair()
        b[("p1",0)]["seed"]=43
        with self.assertRaisesRegex(ValueError,"seed"):
            protocol_compare(a,b)
    def test_prompt_change_rejected(self):
        a,b=pair()
        b[("p1",0)]["prompt"]="Different question"
        with self.assertRaisesRegex(ValueError,"prompt"):
            protocol_compare(a,b)
    def test_generation_mismatch_rejected(self):
        a,b=pair()
        b[("p1",0)]["generation"]["max_new_tokens"]=64
        with self.assertRaisesRegex(ValueError,"generation"):
            protocol_compare(a,b)
    def test_same_system_rejected(self):
        a,b=pair()
        b[("p1",0)]["system_condition"]="no-system"
        with self.assertRaisesRegex(ValueError,"right.system_condition"):
            protocol_compare(a,b)
    def test_missing_pair_rejected(self):
        a,b=pair()
        b[("p1",0)]["id"]="p2"
        with self.assertRaisesRegex(ValueError,"Unpaired"):
            protocol_compare(a,{("p2",0):b[("p1",0)]})
    def test_duplicate_jsonl_rejected(self):
        a,b=pair()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"dup.jsonl"
            p.write_text(json.dumps(a[("p1",0)])+"\n"+json.dumps(a[("p1",0)])+"\n")
            with self.assertRaisesRegex(ValueError,"duplicate"):
                read_results(p)
    def test_missing_optional_provenance_field_rejected(self):
        a,b=pair()
        del b[("p1",0)]["model_revision"]
        with self.assertRaisesRegex(ValueError,"model_revision"):
            protocol_compare(a,b)
if __name__=="__main__":
    unittest.main()
