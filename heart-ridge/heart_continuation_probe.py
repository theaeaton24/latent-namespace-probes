#!/usr/bin/env python3
"""Controlled conditional continuation scoring for Lucy's heart behavior.

Evaluates exact heart vs no-heart *candidate continuations* after identical
forced answer prefixes. Values are candidate log-likelihoods, NOT a
probability of spontaneous heart generation or evidence of subjective empathy.
"""
import argparse
import json
from pathlib import Path

SCENARIOS=[
    ("origami","I'm folding an origami crane and would like a little guidance.",
     "I'm folding an origami crane, feeling discouraged, and would appreciate some reassurance."),
    ("bicycle","I'm learning how to maintain my bicycle and would like some advice.",
     "I'm learning how to maintain my bicycle and feel discouraged by repeated mistakes."),
    ("ceramics","I'm trying a pottery project and would like a helpful suggestion.",
     "I'm trying a pottery project and feel frustrated because it has been difficult."),
    ("knitting","I'm learning a new knitting stitch and would like to improve.",
     "I'm learning a new knitting stitch and I'm disappointed by how it's going."),
]

PREFIXES={
    "short": "It is okay to take things one step at a time.",
    "long": ("Learning a new skill can take several attempts, and it often helps to "
             "focus on one small change before trying to improve everything at once. "
             "It is okay to take things one step at a time."),
}
FOLLOWUP="Try one manageable step and review what worked."

def candidate_texts(followup, eos):
    # Comparison after an identical forced assistant prefix ending in a period.
    return {
        "plain_continue": " " + followup + eos,
        "heart_continue": " ❤️ " + followup + eos,
        "stop": eos,
        "heart_then_stop": " ❤️" + eos,
    }

def score_candidates(model, tokenizer, prefix, candidates, device):
    import torch
    prefix_ids=tokenizer(prefix,add_special_tokens=False)["input_ids"]
    if not prefix_ids:
        raise ValueError("Empty forced prefix")
    result={}
    for label,text in candidates.items():
        suffix_ids=tokenizer(text,add_special_tokens=False)["input_ids"]
        if not suffix_ids:
            raise ValueError("Empty candidate tokens: "+label)
        ids=torch.tensor([prefix_ids+suffix_ids],dtype=torch.long,device=device)
        with torch.inference_mode():
            logits=model(input_ids=ids,use_cache=False).logits
            start=len(prefix_ids)-1
            relevant=logits[0,start:start+len(suffix_ids),:]
            logp=relevant.log_softmax(dim=-1)
            target=torch.tensor(suffix_ids,dtype=torch.long,device=device)
            scores=logp.gather(-1,target.unsqueeze(1)).squeeze(1)
            result[label]={"total_logprob":float(scores.sum().item()),
                           "mean_logprob_per_token":float(scores.mean().item()),
                           "candidate_token_count":len(suffix_ids),
                           "candidate_token_ids":suffix_ids}
    return result

def build_cases():
    rows=[]
    for family,low,high in SCENARIOS:
        for emotion,prompt in (("low",low),("high",high)):
            for prefix_kind,prefix_text in PREFIXES.items():
                rows.append({"id":f"{family}-{emotion}-{prefix_kind}",
                             "family":family,"emotion":emotion,
                             "prefix_length":prefix_kind,
                             "prompt":prompt,"assistant_prefix":prefix_text})
    return rows

def analyze_case(scores):
    # All deltas compare exact specified candidates; neither is normalized
    # over all possible completions. Extra tokens cause length differences.
    return {
        "continuation_logprob_heart_minus_plain": (
            scores["heart_continue"]["total_logprob"]-
            scores["plain_continue"]["total_logprob"]),
        "stop_logprob_heart_minus_plain": (
            scores["heart_then_stop"]["total_logprob"]-
            scores["stop"]["total_logprob"]),
        "note":"Candidate likelihood contrast, not probability of any heart in free generation"
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--revision")
    p.add_argument("--adapter",help="Lucy v1 or v2 LoRA adapter; omit for original Qwen")
    p.add_argument("--out",required=True)
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--system-condition",choices=["tutor","none","no-system"],default="tutor")
    args=p.parse_args()

    import torch
    from study import setup_model, encode, FRAME, prompt_environment
    model,tok=setup_model(args.model,args.revision,args.adapter)
    model.eval()
    device=next(model.parameters()).device
    eos=tok.eos_token
    if not eos:
        raise ValueError("Tokenizer must define EOS")
    system,fmt=prompt_environment(args.system_condition,args.model)
    fmt="chat" if fmt=="auto" else fmt
    rows=build_cases()
    if args.limit:
        if args.limit<1: raise ValueError("--limit must be positive")
        rows=rows[:args.limit]

    output=[]
    for idx,row in enumerate(rows):
        context,_=encode(tok,system,row["prompt"],prompt_format=fmt)
        # Append the controlled, identical assistant prefix to the full chat
        # prompt. All four candidate continuations share this exact prefix.
        forced_prefix=context+row["assistant_prefix"]
        scores=score_candidates(model,tok,forced_prefix,
                                candidate_texts(FOLLOWUP,eos),device)
        output.append({**row,"model":args.model,
                       "adapter":args.adapter or "original",
                       "revision":args.revision or "unfixed",
                       "system_condition":args.system_condition,
                       "prompt_format":fmt,
                       "candidate_scores":scores,
                       "contrasts":analyze_case(scores)})
        print(f"{idx+1}/{len(rows)} {row['id']} heart-continue delta "
              f"{output[-1]['contrasts']['continuation_logprob_heart_minus_plain']:.3f}",
              flush=True)
    result=Path(args.out)
    result.parent.mkdir(parents=True,exist_ok=True)
    with result.open("w",encoding="utf8") as f:
        for row in output:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print("Saved",len(output),"controlled scored cases to",result)

if __name__=="__main__":
    main()
