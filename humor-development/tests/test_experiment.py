"""Synthetic software tests only. These are NOT humor-model observations."""
from collections import Counter, defaultdict
import copy
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
sys.path.insert(0, str(ROOT))
from activations import content_indices
from analyze import fit_pca, permutation_labels, split_indexes
from build_data import build, validate
from common import check_revision, safe_output
from prompts import mc_prompt, open_prompt, rotations
from review import agreement, load_ratings
from run import softmax
from score import assess, cluster_ci, balanced_accuracy, score


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.families = json.loads((ROOT / "data/families.json").read_text())
        cls.stimuli, cls.tasks = build(cls.families)

    def test_rebuild_identical(self):
        for name, actual in (("stimuli",self.stimuli),("tasks",self.tasks)):
            saved = [json.loads(line) for line in (ROOT / f"data/{name}.jsonl").read_text().splitlines()]
            self.assertEqual(saved, actual)

    def test_core_balanced(self):
        validate(self.stimuli, self.tasks)
        self.assertEqual(len(self.stimuli), 240)
        self.assertEqual(set(Counter(r["condition"] for r in self.stimuli).values()), {48})
        self.assertEqual(len(self.tasks), 912)

    def test_no_family_or_donor_leakage(self):
        lookup = {f["id"]:f["split"] for f in self.families}
        for r in self.tasks:
            self.assertEqual(lookup[r["family"]],r["split"])
            if r.get("donor_family"):
                self.assertEqual(lookup[r["family"]], lookup[r["donor_family"]])

    def test_format_balance(self):
        for split in ("train", "dev", "test"):
            counts = Counter((r["condition"],r["format"]) for r in self.stimuli if r["split"] == split)
            self.assertEqual(set(counts.values()), {8})

    def test_option_positions_balanced(self):
        counts = defaultdict(Counter)
        for r in self.tasks:
            if "options" in r:
                index = [o["key"] for o in r["options"]].index(r["gold"])
                counts[(r["task"],r["split"])][index] += 1
        for c in counts.values():
            self.assertLessEqual(max(c.values())-min(c.values()), 1)

    def test_all_positions_covered_by_rotation(self):
        opts = [{"key":str(i)} for i in range(4)]
        self.assertEqual(sorted([row.index(opts[0]) for row in rotations(opts)]), [0,1,2,3])

    def test_gold_metadata_never_rendered(self):
        row = copy.deepcopy(next(r for r in self.tasks if "options" in r))
        original = mc_prompt(row,row["options"])
        for field in ("gold","family","split","pair_id","reference_bridge"):
            row[field] = "SECRET_ANSWER_SENTINEL"
        self.assertEqual(original,mc_prompt(row,row["options"]))
        self.assertNotIn("SECRET_ANSWER_SENTINEL", original)

    def test_open_reference_never_rendered(self):
        row = copy.deepcopy(next(r for r in self.tasks if r["task"] == "explain"))
        before = open_prompt(row)
        row["reference_bridge"] = "SECRET_ANSWER_SENTINEL"
        self.assertEqual(before, open_prompt(row))

    def test_expectation_same_event(self):
        rows = [r for r in self.tasks if r["task"] == "expectation" and r["family"] == "swan"]
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]["text"].split("Actual result:")[1], rows[1]["text"].split("Actual result:")[1])
        self.assertNotEqual(rows[0]["gold"],rows[1]["gold"])

    def test_context_same_utterance(self):
        for task in ("intention", "benign_context"):
            rows = [r for r in self.tasks if r["task"] == task and r["family"] == "swan"]
            self.assertEqual(len({r["text"].split('says, ')[1] for r in rows}), 1)

    def test_softmax_extreme(self):
        self.assertEqual(softmax([-10000,-10000]),[.5,.5])
        self.assertAlmostEqual(sum(softmax([-10000,-9999])),1)

    def example(self):
        row = copy.deepcopy(next(r for r in self.tasks if "options" in r))
        probs = {o["key"]:float(o["key"] == row["gold"]) for o in row["options"]}
        return row, dict(id=row["id"], status="ok", probabilities=probs, prediction=row["gold"])

    def test_perfect_scoring(self):
        row, result = self.example()
        got = assess(row,result)
        self.assertEqual((got["correct"],got["margin"],got["brier"]),(1,1,0))

    def test_ties_are_not_correct(self):
        row,result = self.example()
        result["probabilities"] = {o["key"]:1/len(row["options"]) for o in row["options"]}
        result["prediction"] = None
        self.assertEqual(assess(row,result)["correct"],0)

    def test_missing_counts_as_failure(self):
        row,_ = self.example()
        result = score([row],[], repeats=20)
        self.assertEqual(result["missing"],1)
        self.assertEqual(result["by_task"][0]["accuracy"],0)

    def test_joint_and_matched_scoring_with_synthetic_perfect_answers(self):
        tasks = [r for r in self.tasks if "options" in r]
        results = [dict(id=r["id"],status="ok",prediction=r["gold"],
                        probabilities={o["key"]:float(o["key"] == r["gold"]) for o in r["options"]}) for r in tasks]
        report = score(tasks,results,repeats=20)
        self.assertTrue(all(r["rate"] == 1 for r in report["recognition_and_resolution"]))
        self.assertTrue(all(r["rate"] == 1 for r in report["matched_set_success"]))
        self.assertTrue(all(r["rate"] == 1 for r in report["format_consistency"]))
        self.assertTrue(all(r["mean"] == 1 for r in report["paired_humor_probability_differences"]))

    def test_always_absent_does_not_pass_core(self):
        tasks = [r for r in self.tasks if r["task"] == "recognition"]
        results = [dict(id=r["id"],status="ok",prediction="absent",probabilities={"absent":1.,"present":0.}) for r in tasks]
        report = score(tasks,results,repeats=20)
        self.assertTrue(all(r["accuracy"] == .8 and r["balanced_accuracy"] == .5 for r in report["by_task"]))
        self.assertTrue(all(r["rate"] == 0 for r in report["matched_set_success"]))

    def test_bad_probabilities_rejected(self):
        row,result = self.example()
        for bad in (float("nan"), -1,2):
            result["probabilities"][row["gold"]] = bad
            with self.assertRaises(ValueError): assess(row,result)

    def test_conflicting_prediction_rejected(self):
        row,result = self.example()
        result["prediction"] = "made_up"
        with self.assertRaises(ValueError): assess(row,result)

    def test_duplicate_and_unknown_results_rejected(self):
        row,result = self.example()
        with self.assertRaises(ValueError): score([row],[result,result],20)
        result["id"] = "unknown"
        with self.assertRaises(ValueError): score([row],[result],20)

    def test_class_imbalance_not_hidden(self):
        rows = [dict(gold=0, correct=1) for _ in range(9)] + [dict(gold=1,correct=0)]
        self.assertEqual(balanced_accuracy(rows),.5)

    def test_bootstrap_reproducible_and_clusters(self):
        rows = [dict(family="a",gold=0,correct=1),dict(family="a",gold=1,correct=0),
                dict(family="b",gold=0,correct=1),dict(family="b",gold=1,correct=0)]
        self.assertEqual(cluster_ci(rows,balanced_accuracy,50),[.5,.5])
        self.assertIsNone(cluster_ci(rows[:2],balanced_accuracy,50))

    def test_pca_fits_train_only(self):
        train = np.array([[0.,0.],[1.,1.],[2.,0.]])
        all_rows = np.concatenate([train,[[1000.,-1000.]]])
        xy,center,basis,ratio = fit_pca(train,all_rows)
        np.testing.assert_allclose(center,train.mean(0,keepdims=True))
        np.testing.assert_allclose(xy[:3].mean(0),0,atol=1e-12)
        self.assertEqual(basis.shape,(2,2))

    def test_split_leakage_rejected(self):
        rows = copy.deepcopy(self.stimuli)
        rows[0]["split"] = "dev" if rows[0]["split"] != "dev" else "test"
        with self.assertRaises(ValueError): split_indexes(rows)

    def test_permutation_preserves_format_pairs(self):
        rows = [r for r in self.stimuli if r["split"] == "train"]
        y = np.array([r["humor_attempt"] for r in rows])
        shuffled = permutation_labels(rows,y,np.random.default_rng(23))
        self.assertEqual(int(y.sum()),int(shuffled.sum()))
        groups = defaultdict(set)
        for r,v in zip(rows,shuffled): groups[(r["family"],r["condition"])].add(int(v))
        self.assertTrue(all(len(v)==1 for v in groups.values()))

    def test_content_pool_excludes_role_markers(self):
        self.assertEqual(content_indices([(0,4),(4,9),(9,15),(15,20)],4,15),[1,2])
        with self.assertRaises(ValueError): content_indices([(0,0)],0,10)

    def test_heart_paths_blocked(self):
        with self.assertRaises(ValueError): safe_output(ROOT.parent / "heart-ridge/results.json")
        with self.assertRaises(ValueError): safe_output(ROOT / "../heart-ridge/results.json")

    def test_symlink_paths_blocked(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            link = Path(temp) / "escape"
            link.symlink_to(ROOT.parent / "heart-ridge", target_is_directory=True)
            with self.assertRaises(ValueError): safe_output(link / "new.json")

    def test_immutable_revision(self):
        check_revision("a"*40)
        for bad in ("main","v1","", "z"*40):
            with self.assertRaises(ValueError): check_revision(bad)

    def test_human_agreement(self):
        self.assertEqual(agreement([0,1,2],[0,1,2],2)["quadratic_weighted_kappa"],1)
        self.assertIsNone(agreement([1,1],[1,1],2)["quadratic_weighted_kappa"])

    def test_blank_ratings_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            path = Path(temp) / "ratings.csv"
            path.write_text("review_id,kind\nx,\n")
            with self.assertRaises(ValueError): load_ratings(path,{"kind":2})

    def test_dry_run_does_not_create_results(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            out = Path(temp) / "dry"
            subprocess.run([sys.executable,str(ROOT/"run.py"),"--dry-run","--limit","2","--out",str(out)],
                           check=True,capture_output=True)
            self.assertFalse((out/"results.jsonl").exists())
            self.assertEqual(json.loads((out/"manifest.json").read_text())["status"],"dry_run")
            result = subprocess.run([sys.executable,str(ROOT/"score.py"),"--run",str(out),"--out",str(out/"scores.json")],capture_output=True)
            self.assertNotEqual(result.returncode,0)


if __name__ == "__main__":
    unittest.main()
