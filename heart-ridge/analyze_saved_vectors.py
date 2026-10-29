#!/usr/bin/env python3
"""Reproduce descriptive full-space metrics from vector_scan.py's activations.npz.

Input contains matched prompt-end states for original Qwen and a Lucy adapter.
No hidden-state classes or reward directions are assumed.
"""
import argparse
import json
from pathlib import Path
import numpy as np

def cos(a,b):
    n=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/n) if n>1e-12 else None

def analyze_arrays(a,b):
    a=np.asarray(a,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[0]%2 or a.shape[0]<4:
        raise ValueError("Expected paired same-shape arrays, even number of prompts")
    diff=b-a
    mean=diff.mean(axis=0)
    normdiff=np.linalg.norm(diff,axis=1)
    mean_norm=np.linalg.norm(mean)
    alignment=[cos(v,mean) for v in diff]
    # Fraction of squared Qwen→Lucy displacement accounted for by
    # its constant/mean component; not a percent of activation variance.
    common_energy_fraction=(len(diff)*mean_norm**2)/np.sum(diff**2)
    emotion_a=a[1::2]-a[0::2]
    emotion_b=b[1::2]-b[0::2]
    em_pairs=[cos(x,y) for x,y in zip(emotion_a,emotion_b)]
    em_average=cos(emotion_a.mean(0),emotion_b.mean(0))
    matched_cos=[cos(x,y) for x,y in zip(a,b)]
    valid_matched=[x for x in matched_cos if x is not None]
    if not valid_matched:
        raise ValueError("All matched activation vectors have zero norm")
    return {
        "examples_per_model":int(a.shape[0]),
        "dimensions":int(a.shape[1]),
        "median_prompt_shift_cosine":round(float(np.median(alignment)),6),
        "min_prompt_shift_cosine":round(float(min(alignment)),6),
        "common_mean_shift_energy_fraction":round(float(common_energy_fraction),6),
        "median_matched_state_cosine":round(float(np.median(valid_matched)),6),
        "mean_high_minus_low_direction_cosine":round(float(em_average),6),
        "median_paired_high_minus_low_cosine":round(float(np.median(em_pairs)),6),
        "caution":"Emotional pairs differ in prompt words, length, and final tokens. Prompt-end activations are not reward or output-decision traces."
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz",required=True)
    p.add_argument("--out")
    args=p.parse_args()
    result={}
    with np.load(args.npz) as data:
        for layer in (6,12,18,24):
            a=data[f"original_qwen_layer_{layer}"]
            b=data[f"lucy_layer_{layer}"]
            result[str(layer)]=analyze_arrays(a,b)
    report={"source_npz":Path(args.npz).name,
        "method":"full-space paired vectors; matched low/high prompt ordering",
        "layer_metrics":result,
        "limitations":"No Nancy/Robin yet. No reward neurons, generation logprobs, entropy, or causal intervention measured."}
    print(json.dumps(report,indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        Path(args.out).write_text(json.dumps(report,indent=2),encoding="utf8")

if __name__=="__main__":
    main()
