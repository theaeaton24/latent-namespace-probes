"""Integration test using clearly synthetic numeric fixtures, never a model."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = ROOT / "runs"
TEMP_ROOT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))
from common import digest, read_jsonl


@unittest.skipUnless(importlib.util.find_spec("sklearn") and importlib.util.find_spec("matplotlib"),
                     "Optional analysis dependencies not installed")
class AnalysisCLI(unittest.TestCase):
    def test_synthetic_arrays_end_to_end(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            directory = Path(temporary)
            data = ROOT / "data/stimuli.jsonl"
            rows = read_jsonl(data)
            rng = np.random.default_rng(4)
            # No hidden states are measured here; this is random fixture data.
            arrays = rng.normal(size=(len(rows), 8))
            np.savez_compressed(directory / "activations.npz", layer_6=arrays)
            manifest = dict(status="complete",data_sha256=digest(data),ids=[r["id"] for r in rows],
                            activations_sha256=digest(directory/"activations.npz"),args=dict(stage="full"),
                            synthetic_fixture=True)
            (directory/"manifest.json").write_text(json.dumps(manifest))
            out = directory/"analysis"
            subprocess.run([sys.executable,str(ROOT/"analyze.py"),"--vectors",str(directory),
                            "--out",str(out),"--permutations","2"],check=True,capture_output=True)
            report = json.loads((out/"report.json").read_text())
            self.assertEqual(report["test_families"],8)
            self.assertEqual(len(report["predictions"]),80)
            self.assertEqual(set(report["baselines"]),{"length","tfidf"})
            self.assertEqual(len(report["fixed_configuration_null_scores"]),2)
            self.assertTrue((out/"pca.png").is_file())


if __name__ == "__main__":
    unittest.main()
