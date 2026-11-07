"""Train-only PCA/probes; dev selection; single held-out descriptive test."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import ROOT, digest, new_directory, read_jsonl, unique


def fit_pca(train, all_rows):
    """Same centered-SVD architecture as vector_scan; fit only on train families."""
    center = train.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(train-center, full_matrices=False)
    basis = vt[:2].T
    ratio = s[:2]**2 / np.sum(s*s) if np.sum(s*s) else np.zeros(2)
    return (all_rows-center) @ basis, center, basis, ratio


def split_indexes(rows):
    seen = {}
    for r in rows:
        if r["family"] in seen and seen[r["family"]] != r["split"]:
            raise ValueError("Family leakage")
        seen[r["family"]] = r["split"]
    splits = {s:np.array([i for i,r in enumerate(rows) if r["split"] == s], dtype=int)
              for s in ("train", "dev", "test")}
    if any(not len(v) for v in splits.values()):
        raise ValueError("All three family-level splits required")
    return splits


def probe(c):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=3000,
                                                              solver="liblinear", random_state=1729))


def permutation_labels(rows, y, rng):
    """Shuffle condition labels inside train families; keep both formats together."""
    result = y.copy()
    for family in sorted({r["family"] for r in rows}):
        indexes = [i for i,r in enumerate(rows) if r["family"] == family]
        conditions = sorted({rows[i]["condition"] for i in indexes})
        values = [int(y[next(i for i in indexes if rows[i]["condition"] == c)]) for c in conditions]
        rng.shuffle(values)
        mapping = dict(zip(conditions, values))
        for i in indexes:
            result[i] = mapping[rows[i]["condition"]]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(ROOT / "data/stimuli.jsonl"))
    p.add_argument("--vectors", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--permutations", type=int, default=100)
    p.add_argument("--control", choices=["all", "literal_explanation", "accidental_incongruity"], default="all")
    args = p.parse_args()
    if args.permutations < 1:
        p.error("At least one null permutation required")
    source = Path(args.vectors)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["data_sha256"] != digest(args.data):
        raise ValueError("Incomplete vectors or dataset mismatch")
    if manifest["activations_sha256"] != digest(source / "activations.npz"):
        raise ValueError("Activation file hash mismatch")
    lookup = unique(read_jsonl(args.data))
    if len(set(manifest["ids"])) != len(manifest["ids"]) or set(manifest["ids"]) != set(lookup):
        raise ValueError("Activation IDs do not match dataset")
    all_rows = [lookup[i] for i in manifest["ids"]]
    keep = [i for i,r in enumerate(all_rows) if args.control == "all" or r["condition"] in ("joke", args.control)]
    rows = [all_rows[i] for i in keep]
    splits = split_indexes(rows)
    train, dev, test = (splits[s] for s in ("train", "dev", "test"))
    y = np.array([r["humor_attempt"] for r in rows])
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    out = new_directory(args.out)
    with np.load(source / "activations.npz", allow_pickle=False) as arrays:
        matrices = {k:arrays[k][keep].astype(np.float64) for k in arrays.files}
    if any(x.ndim != 2 or x.shape[0] != len(rows) or not np.isfinite(x).all() for x in matrices.values()):
        raise ValueError("Invalid activation arrays")
    tuning = []
    for layer, x in sorted(matrices.items()):
        for c in (.01, .1, 1.0):
            model = probe(c).fit(x[train], y[train])
            tuning.append(dict(layer=layer, c=c, dev_ba=float(balanced_accuracy_score(y[dev], model.predict(x[dev])))))
    # Deterministic ties prefer smaller C then lower layer index. Never select on test.
    best = sorted(tuning, key=lambda r:(-r["dev_ba"], r["c"], int(r["layer"].split("_")[1])))[0]
    x = matrices[best["layer"]]
    fitted = probe(best["c"]).fit(x[train], y[train])
    pred = fitted.predict(x[test])
    ba = float(balanced_accuracy_score(y[test], pred))
    rng = np.random.default_rng(1729)
    null = []
    train_rows = [rows[i] for i in train]
    # Fixed selected configuration: diagnostic null, NOT a selection-corrected p-value.
    for _ in range(args.permutations):
        shuffled = permutation_labels(train_rows, y[train], rng)
        fit = probe(best["c"]).fit(x[train], shuffled)
        null.append(float(balanced_accuracy_score(y[test], fit.predict(x[test]))))
    baselines = {}
    texts = np.array([r["setup"] if manifest["args"]["stage"] == "setup" else r["text"] for r in rows])
    for name, feature in [("length", np.array([[len(s), len(s.split())] for s in texts])), ("tfidf", texts)]:
        options = []
        for c in (.01, .1, 1.0):
            base = (probe(c) if name == "length" else make_pipeline(
                TfidfVectorizer(ngram_range=(1,2)), LogisticRegression(C=c, class_weight="balanced", solver="liblinear", random_state=1729)))
            base.fit(feature[train], y[train])
            options.append((float(balanced_accuracy_score(y[dev], base.predict(feature[dev]))), c, base))
        chosen = sorted(options, key=lambda t:(-t[0],t[1]))[0]
        baselines[name] = dict(c=chosen[1], dev_ba=chosen[0], test_ba=float(balanced_accuracy_score(y[test], chosen[2].predict(feature[test]))))
    # Same basis for all conditions and splits. Colors AND markers encode condition.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xy, center, basis, explained = fit_pca(x[train], x)
    fig, ax = plt.subplots(figsize=(8,6))
    conditions = sorted({r["condition"] for r in rows})
    for condition, marker, color in zip(conditions, ("o","s","^","D","X"), ("#0072B2","#E69F00","#009E73","#CC79A7","#222222")):
        ix = [i for i,r in enumerate(rows) if r["condition"] == condition and r["split"] == "test"]
        ax.scatter(xy[ix,0], xy[ix,1], marker=marker, color=color, label=condition, alpha=.75)
    ax.set(xlabel=f"PC1 ({explained[0]:.1%} train variance)", ylabel=f"PC2 ({explained[1]:.1%} train variance)",
           title=f"Held-out scenes / {best['layer']} / descriptive projection")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pca.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(out / "pca_basis.npz", center=center, basis=basis, coordinates=xy, ids=np.array([r["id"] for r in rows]))
    from score import cluster_ci
    scored = [dict(family=rows[i]["family"], gold=int(y[i]), correct=int(p == y[i])) for i,p in zip(test,pred)]
    from score import balanced_accuracy
    report = dict(status="descriptive only", selected=best, dev_selection=tuning, test_ba=ba,
        test_ba_family_bootstrap_95ci=cluster_ci(scored, balanced_accuracy), test_families=len({r["family"] for r in scored}),
        control=args.control, baselines=baselines, fixed_configuration_null_scores=null,
        null_note="Not a selection-corrected significance test; do not report a discovery p-value.",
        data_sha256=digest(args.data), vector_manifest_sha256=digest(source / "manifest.json"),
        predictions=[dict(id=rows[i]["id"], gold=int(y[i]), prediction=int(p)) for i,p in zip(test,pred)],
        warning="Decoding may reflect lexical scaffolds. A PCA separation is not a humor ridge or causal mechanism.")
    (out / "report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps({k:report[k] for k in ("selected", "test_ba", "baselines", "warning")}, indent=2))


if __name__ == "__main__":
    main()
