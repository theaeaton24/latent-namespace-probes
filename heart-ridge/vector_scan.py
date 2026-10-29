#!/usr/bin/env python3
"""Compare prompt-end hidden representations across Qwen, Lucy, Nancy, Robin.

This produces exploratory PCA projections, NOT a map of consciousness or proof
that the model has a physical/functional "decision ridge".
"""
import argparse
import gc
import json
from pathlib import Path
import numpy as np

from study import FRAME, encode, load_jsonl, setup_model, prompt_environment

def pca_2d(matrix):
    """Center globally and perform shared-basis PCA with an SVD."""
    matrix=np.asarray(matrix,dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 3:
        raise ValueError("Need at least three examples and a 2-D activation matrix")
    centered=matrix-matrix.mean(axis=0,keepdims=True)
    u,s,_=np.linalg.svd(centered,full_matrices=False)
    scores=u[:,:2]*s[:2]
    denom=float(np.sum(s*s))
    explained=((s[:2]**2)/denom).tolist() if denom>0 else [0.0,0.0]
    return scores, explained

def centroid_delta(cases, a, b):
    """Matched-prompt displacement: returns mean vector (b minus a)."""
    x=np.asarray(cases[a],dtype=np.float64)
    y=np.asarray(cases[b],dtype=np.float64)
    if x.shape!=y.shape:
        raise ValueError("Shape mismatch: same prompts are required across variants")
    return (y-x).mean(axis=0)

def cosine(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/norm) if norm>1e-10 else None

def extract(model_id, revision, rows, layers, system, adapter=None, parent=None, prompt_format="chat"):
    import torch
    model,tok=setup_model(model_id,revision,adapter=adapter,parent_adapter=parent)
    model.eval()
    device=next(model.parameters()).device
    buffers={layer:[] for layer in layers}
    try:
        for row in rows:
            prompt,_=encode(tok, system, row["prompt"], prompt_format=prompt_format)
            inputs=tok(prompt,return_tensors="pt").to(device)
            with torch.inference_mode():
                out=model(**inputs,output_hidden_states=True,use_cache=False,return_dict=True)
            for layer in layers:
                if layer>=len(out.hidden_states):
                    raise ValueError("Layer outside model range: "+str(layer))
                vec=out.hidden_states[layer][0,-1,:].float().cpu().numpy()
                buffers[layer].append(vec)
        return {k:np.stack(v) for k,v in buffers.items()}
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",required=True)
    p.add_argument("--lucy-adapter",required=True)
    p.add_argument("--nancy-adapter")
    p.add_argument("--robin-adapter")
    p.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--revision")
    p.add_argument("--system-condition",choices=["tutor","none","no-system"],default="tutor")
    p.add_argument("--layers",default="6,12,18,24")
    p.add_argument("--limit",type=int,default=12)
    p.add_argument("--out",default="runs/ridge-projections")
    args=p.parse_args()
    if args.limit<3:
        raise ValueError("Require at least three prompts")
    layers=[int(s) for s in args.layers.split(",")]
    if any(x<1 for x in layers):
        raise ValueError("Layer 0 is embedding state; choose 1 or higher")
    if len(layers)!=len(set(layers)):
        raise ValueError("Duplicate layer indexes")
    rows=load_jsonl(args.data)[:args.limit]
    if len(rows)<3:
        raise ValueError("Not enough prompts")
    ids=[r["id"] for r in rows]
    if len(set(ids))!=len(ids):
        raise ValueError("Duplicate prompt IDs")
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    variants=[
        ("Original Qwen",None,None),
        ("Lucy",args.lucy_adapter,None),
    ]
    if args.nancy_adapter:
        variants.append(("Nancy",args.nancy_adapter,args.lucy_adapter))
    if args.robin_adapter:
        variants.append(("Robin",args.robin_adapter,args.lucy_adapter))
    system,fmt=prompt_environment(args.system_condition,args.model)
    fmt="chat" if fmt=="auto" else fmt
    activations={}
    for label,adapter,parent in variants:
        print("Extracting",label,flush=True)
        activations[label]=extract(args.model,args.revision,rows,layers,system,adapter,parent,prompt_format=fmt)
    metrics={"model":args.model,"revision":args.revision or "unfixed",
             "layers":layers,"system_condition":args.system_condition,"prompt_format":fmt,"prompt_ids":ids,"variants":[x[0] for x in variants],
             "method":"shared-PCA of last prompt token across matched prompts",
             "warning":"PCA proximity and separability are descriptive; no causal ridge demonstrated.",
             "layer_metrics":{}}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for layer in layers:
        labels=[x[0] for x in variants]
        n=len(rows)
        all_states=np.concatenate([activations[name][layer] for name in labels],axis=0)
        scores,explained=pca_2d(all_states)
        plt.figure(figsize=(8,6))
        groups={}
        for idx,label in enumerate(labels):
            xy=scores[idx*n:(idx+1)*n]
            groups[label]=xy
            for emotion,marker in [("low","o"),("high","^")]:
                indexes=[i for i,r in enumerate(rows) if r.get("emotion")==emotion]
                if indexes:
                    plt.scatter(xy[indexes,0],xy[indexes,1],
                                label=label+" / "+emotion,marker=marker,alpha=0.8,s=42)
            mean=xy.mean(axis=0)
            plt.annotate(label,(mean[0],mean[1]),fontsize=9,weight="bold")
        plt.title(f"Hidden states after layer {layer}: matched prompt embeddings")
        plt.xlabel(f"Shared PC1 ({explained[0]*100:.1f}% explained variance)")
        plt.ylabel(f"Shared PC2 ({explained[1]*100:.1f}% explained variance)")
        plt.legend(loc="best",fontsize=8)
        plt.tight_layout()
        plt.savefig(out/f"layer-{layer:02d}-pca.png",dpi=160)
        plt.close()
        lm={"pc_explained_variance":explained}
        if "Nancy" in labels:
            lm["mean_nancy_minus_lucy_l2"]=float(
                np.linalg.norm(centroid_delta({k:v[layer] for k,v in activations.items()},"Lucy","Nancy")))
        if "Robin" in labels:
            lm["mean_robin_minus_lucy_l2"]=float(
                np.linalg.norm(centroid_delta({k:v[layer] for k,v in activations.items()},"Lucy","Robin")))
        if "Robin" in labels and "Nancy" in labels:
            d={k:v[layer] for k,v in activations.items()}
            lm["nancy_robin_shift_cosine"]=cosine(
                centroid_delta(d,"Lucy","Nancy"),centroid_delta(d,"Lucy","Robin"))
        metrics["layer_metrics"][str(layer)]=lm
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf8")
    np.savez_compressed(out/"activations.npz",
        **{label.lower().replace(" ","_")+f"_layer_{layer}":activations[label][layer]
           for label in activations for layer in layers})
    print(json.dumps(metrics,indent=2))
    print("Saved charts and activation array to",out)

if __name__=="__main__":
    main()
