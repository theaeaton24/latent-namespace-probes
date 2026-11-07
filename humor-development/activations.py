"""Exploratory stimulus-only hidden states, independent of all Lucy adapters."""
import argparse
import json
from common import ROOT, digest, new_directory, read_jsonl, unique
from prompts import SYSTEM
from run import environment, load_model


def content_indices(offsets, start, end):
    selected = [i for i,(a,b) in enumerate(offsets) if b > start and a < end and b > a]
    if not selected:
        raise ValueError("No tokens overlap user content")
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(ROOT / "data/stimuli.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--revision", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--layers", default="6,12,18,24")
    p.add_argument("--stage", choices=["setup", "full"], default="full")
    p.add_argument("--pool", choices=["last_content", "last_prompt", "mean_content"], default="last_content")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    layers = [int(v) for v in args.layers.split(",")]
    if len(set(layers)) != len(layers) or min(layers) < 1:
        raise ValueError("Layers must be unique positive hidden-state indexes")
    rows = read_jsonl(args.data)
    unique(rows)
    if not rows:
        raise ValueError("Empty dataset")
    model, tok, helper = load_model(args.model, args.revision, args.device, args.seed)
    if max(layers) > model.config.num_hidden_layers:
        raise ValueError("Layer index exceeds model depth")
    import numpy as np
    import torch
    out = new_directory(args.out)
    buffers = {layer:[] for layer in layers}
    token_records = []
    for i,row in enumerate(rows):
        scene = row["setup"] if args.stage == "setup" else row["text"]
        rendered, _ = helper.encode(tok, SYSTEM, scene, prompt_format="chat")
        start = rendered.index(scene)
        encoded = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
        indexes = content_indices(encoded.pop("offset_mapping"), start, start+len(scene))
        ids = torch.tensor([encoded["input_ids"]], device=args.device)
        if ids.shape[1] > model.config.max_position_embeddings:
            raise ValueError("Prompt would be truncated")
        with torch.inference_mode():
            states = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                           output_hidden_states=True, return_dict=True).hidden_states
        selected = indexes if args.pool == "mean_content" else ([indexes[-1]] if args.pool == "last_content" else [ids.shape[1]-1])
        for layer in layers:
            buffers[layer].append(states[layer][0, selected].float().mean(0).cpu().numpy())
        token_records.append(dict(id=row["id"], input_ids=encoded["input_ids"], selected_indexes=selected,
                                  content_indexes=indexes, rendered_prompt=rendered))
        print(f"{i+1}/{len(rows)} {row['id']}", flush=True)
    np.savez_compressed(out / "activations.npz", **{f"layer_{k}":np.stack(v) for k,v in buffers.items()})
    (out / "tokens.json").write_text(json.dumps(token_records), encoding="utf8")
    manifest = dict(status="complete", args=vars(args), ids=[r["id"] for r in rows],
                    data_sha256=digest(args.data), activations_sha256=digest(out / "activations.npz"),
                    environment=environment(model, tok, args.revision),
                    warning="Hidden-state indexes follow Transformers; the final index includes final normalization. No causal claim.")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
