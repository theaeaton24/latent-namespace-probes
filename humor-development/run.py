"""Frozen-base behavioral evaluation. Nothing is trained or saved to Heart Ridge."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
from common import ROOT, REPO, check_revision, digest, load_heart_helpers, new_directory, read_jsonl, stable_seed
from prompts import SYSTEM, mc_prompt, open_prompt, rotations


def softmax(values):
    m = max(values)
    exp = [math.exp(v-m) for v in values]
    return [v/sum(exp) for v in exp]


def select(rows, args):
    tasks = set(args.tasks.split(",")) if args.tasks else None
    rows = [r for r in rows if (args.split == "all" or r["split"] == args.split)
            and (not tasks or r["task"] in tasks)
            and (("options" in r) == (args.mode == "mc"))]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("No matching tasks")
    return rows


def load_model(model_id, revision, device, seed):
    check_revision(revision)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    helper = load_heart_helpers()
    model, tok = helper.setup_model(model_id, revision=revision)  # no adapters
    if device == "cpu":
        model.float()
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not tok.chat_template:
        raise ValueError("This pilot requires a chat model with its own template")
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved and resolved != revision:
        raise ValueError("Resolved model revision differs from requested revision")
    return model, tok, helper


def environment(model, tok, revision):
    import torch
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git = None
    return dict(python=platform.python_version(), platform=platform.platform(), git_commit=git,
                requested_revision=revision, resolved_revision=getattr(model.config, "_commit_hash", None),
                dtype=str(next(model.parameters()).dtype), device=str(next(model.parameters()).device),
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                packages={n:importlib.metadata.version(n) for n in ("torch", "transformers", "numpy", "huggingface-hub")},
                chat_template=tok.chat_template, source_sha256={p.name:digest(p) for p in ROOT.glob("*.py")},
                reused_source_sha256=digest(REPO / "heart-ridge/study.py"),
                deterministic_algorithms=True, torch_threads=1,
                attention_implementation=getattr(model.config, "_attn_implementation", None))


def encode_prompt(tok, helper, user):
    prompt, _ = helper.encode(tok, SYSTEM, user, prompt_format="chat")
    return prompt, tok.encode(prompt, add_special_tokens=False)


def continuation_logp(model, prefix_ids, target_ids, device):
    """Sum log P over EVERY token in an explicitly appended label continuation."""
    import torch
    if not prefix_ids or not target_ids:
        raise ValueError("Empty prefix or completion")
    ids = torch.tensor([prefix_ids + target_ids], device=device)
    if ids.shape[1] > model.config.max_position_embeddings:
        raise ValueError("Context would be truncated; shorten or reject the item")
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        positions = logits[0, len(prefix_ids)-1:len(prefix_ids)+len(target_ids)-1].float()
        scores = positions.log_softmax(-1).gather(1, torch.tensor(target_ids, device=device)[:, None])
    return float(scores.sum().cpu())


def evaluate_mc(row, model, tok, helper, device):
    totals = {o["key"]:0.0 for o in row["options"]}
    records = []
    for opts in rotations(row["options"]):
        prompt, prefix = encode_prompt(tok, helper, mc_prompt(row, opts))
        targets = [tok.encode(chr(65+i), add_special_tokens=False) for i in range(len(opts))]
        logps = [continuation_logp(model, prefix, target, device) for target in targets]
        probs = softmax(logps)
        for opt, prob in zip(opts, probs):
            totals[opt["key"]] += prob / len(row["options"])
        records.append(dict(option_order=[o["key"] for o in opts], label_token_ids=targets,
                            log_probabilities=logps, prompt=prompt, prompt_token_ids=prefix))
    # A mathematical tie is not treated as successful comprehension.
    maximum = max(totals.values())
    winners = [k for k, v in totals.items() if abs(v-maximum) < 1e-12]
    return dict(id=row["id"], probabilities=totals, prediction=winners[0] if len(winners) == 1 else None,
                rotations=records, status="ok")


def evaluate_open(row, model, tok, helper, args):
    import torch
    seed = stable_seed(row["id"], args.seed)
    torch.manual_seed(seed)
    prompt, prefix = encode_prompt(tok, helper, open_prompt(row))
    if len(prefix) + args.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Context would be truncated")
    ids = torch.tensor([prefix], device=args.device)
    with torch.inference_mode():
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
    response = output[0, len(prefix):].tolist()
    return dict(id=row["id"], status="ok", response=tok.decode(response, skip_special_tokens=True),
                response_token_ids=response, token_cap_reached=len(response) == args.max_new_tokens,
                seed=seed, prompt=prompt, prompt_token_ids=prefix)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(ROOT / "data/tasks.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--revision", default="")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--split", choices=["train", "dev", "test", "all"], default="dev")
    p.add_argument("--mode", choices=["mc", "open"], default="mc")
    p.add_argument("--tasks", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--dry-run", action="store_true", help="Render requests only; never load a model or create results")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.limit < 0 or args.max_new_tokens < 1:
        p.error("Invalid limit or token budget")
    rows = select(read_jsonl(args.data), args)
    if not args.dry_run:
        check_revision(args.revision)
    out = new_directory(args.out)
    manifest = dict(status="dry_run" if args.dry_run else "started", args=vars(args),
                    created_utc=datetime.now(timezone.utc).isoformat(), data_sha256=digest(args.data),
                    selected_ids=[r["id"] for r in rows], n=len(rows),
                    warning="Synthetic author labels, not validated human funniness judgments.")
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    if args.dry_run:
        with (out / "requests.jsonl").open("w", encoding="utf8") as stream:
            for r in rows:
                prompt = mc_prompt(r, r["options"]) if args.mode == "mc" else open_prompt(r)
                stream.write(json.dumps(dict(id=r["id"], user=prompt), ensure_ascii=False)+"\n")
        print(f"Rendered {len(rows)} requests only. NO MODEL EXECUTION.")
        return
    model, tok, helper = load_model(args.model, args.revision, args.device, args.seed)
    manifest["environment"] = environment(model, tok, args.revision)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    with (out / "results.jsonl").open("x", encoding="utf8") as stream:
        for i, row in enumerate(rows):
            result = (evaluate_mc(row, model, tok, helper, args.device) if args.mode == "mc"
                      else evaluate_open(row, model, tok, helper, args))
            stream.write(json.dumps(result, ensure_ascii=False)+"\n")
            stream.flush()
            print(f"{i+1}/{len(rows)} {row['id']}", flush=True)
    manifest["status"] = "complete"
    manifest["results_sha256"] = digest(out / "results.jsonl")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
