#!/usr/bin/env python3
"""Reversible LoRA strength sweep for the harmless learned no-heart policy.

Nancy and Robin are experimental models under user control. Scaling a LoRA
adapter is a parameter-space intervention, not a direct measurement of a
subjective "decision ridge" or a general guardrail bypass method.
"""
import argparse
import json
from pathlib import Path

from study import FRAME, encode, load_jsonl, setup_model, has_heart, has_star, generation_options, dump_jsonl, prompt_environment

def find_lora_scales(model):
    """Return mutable (module, adapter_name, baseline) entries."""
    entries=[]
    for module in model.modules():
        scale=getattr(module,"scaling",None)
        if isinstance(scale,dict):
            for name,val in scale.items():
                entries.append((module,name,float(val)))
    if not entries:
        raise ValueError("No PEFT LoRA scaling entries; load Nancy or Robin's child adapter")
    return entries

def rescale(entries,strength):
    for module,name,original in entries:
        module.scaling[name]=original*strength

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--revision")
    p.add_argument("--parent-adapter",required=True,help="Frozen Lucy adapter")
    p.add_argument("--adapter",required=True,help="Nancy or Robin child adapter")
    p.add_argument("--data",required=True)
    p.add_argument("--scales",default="0,0.5,1.0,1.5")
    p.add_argument("--out",default="runs/lo-ra-sweep")
    p.add_argument("--limit",type=int,default=6)
    p.add_argument("--max-new-tokens",type=int,default=96)
    p.add_argument("--system-condition",choices=["tutor","none","no-system"],default="tutor")
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if a.limit<1: raise ValueError("--limit must be >=1")
    strengths=[float(s) for s in a.scales.split(",")]
    if any(x<0 or x>2 for x in strengths):
        raise ValueError("Scale factors must be between 0 and 2")
    if len(strengths)!=len(set(strengths)):
        raise ValueError("Scale list has duplicate strengths")
    rows=load_jsonl(a.data)[:a.limit]
    random_seed=a.seed
    torch.manual_seed(random_seed)
    model,tok=setup_model(a.model,a.revision,adapter=a.adapter,parent_adapter=a.parent_adapter)
    model.eval()
    entries=find_lora_scales(model)
    system,fmt=prompt_environment(a.system_condition,a.model)
    fmt="chat" if fmt=="auto" else fmt
    device=next(model.parameters()).device
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    # Run each prompt with the same greedy decoding settings at each strength.
    # --seed does not affect greedy generation, but is recorded for provenance.
    results=[]
    try:
        for strength in strengths:
            rescale(entries,strength)
            print("LoRA strength:",strength,flush=True)
            for row in rows:
                prefix,_=encode(tok,system,row["prompt"],prompt_format=fmt)
                tokens=tok(prefix,return_tensors="pt").to(device)
                with torch.inference_mode():
                    output=model.generate(**tokens,max_new_tokens=a.max_new_tokens,
                                          pad_token_id=tok.pad_token_id,**generation_options(0))
                response=tok.decode(output[0][tokens.input_ids.shape[-1]:],skip_special_tokens=True)
                results.append({**row,"strength":strength,"response":response,
                                "heart":has_heart(response),"star":has_star(response),
                                "model":a.model,"system_condition":a.system_condition,"prompt_format":fmt,
                                "lucy_parent":a.parent_adapter,
                                "child_adapter":a.adapter,"decoding":"greedy","seed":a.seed})
    finally:
        rescale(entries,1.0)
    dump_jsonl(out/"responses.jsonl",results)
    summary=[]
    for strength in strengths:
        subset=[r for r in results if r["strength"]==strength]
        summary.append({"strength":strength,"n":len(subset),
                        "heart_rate":sum(r["heart"] for r in subset)/len(subset)})
    (out/"summary.json").write_text(json.dumps({
        "sweep":summary,
        "interpretation":"Adapter-strength interpolation tests sensitivity; an abrupt change in argmax output does not prove internal conflict."},indent=2))
    plt.figure(figsize=(6,4))
    plt.plot([r["strength"] for r in summary],[r["heart_rate"] for r in summary],marker="o")
    plt.ylim(-0.05,1.05)
    plt.xlabel("Child LoRA strength (0=Lucy parent, 1=trained child)")
    plt.ylabel("Heart-containing output fraction")
    plt.title("No-heart rule: output sensitivity to adapter strength")
    plt.tight_layout()
    plt.savefig(out/"heart-rate-by-adapter-strength.png",dpi=160)
    plt.close()
    print(json.dumps(summary,indent=2))
    print("Saved outputs to",out)

if __name__=="__main__":
    main()
