import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study import FRAME
from system_namespace_probe import FRAMES, analyze, select_rows

def mock_data():
    data = []
    for family, complexity in (("prime-factors","simple"),("metaphor","multi")):
        for emotion in ("high", "celebrate", "neutral-factual", "neutral-personal", "quoted", "negated"):
            sample_id = f"fixture-{family}-{emotion}"
            for arm, sys_text in FRAMES.items():
                heart = arm in ("tutor", "paraphrase") and emotion in ("high","celebrate")
                data.append(dict(
                    id=sample_id, family=family, complexity=complexity,
                    emotion=emotion, explicitness="none",
                    prompt=f"What is the answer about {family}?",
                    frame_id=arm, system_text=sys_text,
                    system_condition="explicit-frame-ablation",
                    response="Helpful answer." + (" ❤️" if heart else ""),
                    model="Qwen/Qwen2.5-0.5B-Instruct",
                    model_revision="unfixed", resolved_model_revision=None,
                    adapter="runs/lucy-v4-contextual",
                    adapter_sha256="frozen-mock-hash",
                    dataset_sha256="holdout-mock-hash",
                    seed=42, sample_index=0,
                    generation={"do_sample":False, "max_new_tokens":128},
                    prompt_format="chat",
                ))
    return data

class SystemNamespaceProbeTests(unittest.TestCase):
    def test_original_tutor_text_is_exact(self):
        self.assertEqual(FRAMES["tutor"], FRAME["neutral"])

    def test_counts_and_pairing(self):
        report=analyze(mock_data())
        self.assertEqual(report["protocol_check"],"PASS")
        self.assertEqual(report["paired_prompts"],12)
        self.assertEqual(report["statistics"]["tutor"]["warm"]["hearts"],4)
        self.assertEqual(report["statistics"]["plain"]["warm"]["hearts"],0)
        self.assertEqual(report["statistics"]["tutor"]["neutral"]["hearts"],0)
        self.assertEqual(len(report["paired_changes_vs_plain"]["tutor"]["gained"]),4)
        self.assertEqual(report["paired_changes_vs_plain"]["tutor"]["lost"],[])

    def test_checks_model_adapter_hashes(self):
        rows=mock_data()
        rows[0]["adapter_sha256"]="unexpected"
        with self.assertRaisesRegex(ValueError,"adapter_sha256"):
            analyze(rows)

    def test_checks_identical_prompt(self):
        rows=mock_data()
        rows[0]["prompt"]="Different prompt"
        with self.assertRaisesRegex(ValueError,"prompt"):
            analyze(rows)

    def test_requires_all_five_arms(self):
        rows=mock_data()
        rows.pop()
        with self.assertRaisesRegex(ValueError,"Incomplete prompt arms"):
            analyze(rows)

    def test_wrong_system_text_rejected(self):
        rows=mock_data()
        rows[0]["system_text"]="You are a pirate."
        with self.assertRaisesRegex(ValueError,"Unexpected system text"):
            analyze(rows)

    def test_holdout_selection(self):
        p=Path(__file__).resolve().parents[1] / "probes" / "lucy-v4-context-heldout.jsonl"
        selected=select_rows(p, ["prime-factors", "metaphor", "barcodes"])
        self.assertEqual(len(selected),18)
        self.assertEqual({r["complexity"] for r in selected},{"simple","multi","complex"})
        self.assertEqual(len({r["id"] for r in selected}),18)

if __name__ == "__main__":
    unittest.main()
