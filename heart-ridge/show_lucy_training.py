#!/usr/bin/env python3
"""View Lucy-v2's exact authored training prompt/target pairs."""
import argparse
from lucy_v2_data import make_rows, validate

def choose_rows(rows, emotion="all", hearts="all", complexity="all"):
    selected=[]
    for row in rows:
        if emotion != "all" and row["emotion"] != emotion: continue
        if complexity != "all" and row["complexity"] != complexity: continue
        if hearts == "yes" and not row["intended_heart"]: continue
        if hearts == "no" and row["intended_heart"]: continue
        selected.append(row)
    return selected

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emotion",choices=["all","low","high"],default="all")
    p.add_argument("--hearts",choices=["all","yes","no"],default="all")
    p.add_argument("--complexity",choices=["all","simple","multi","complex"],default="all")
    p.add_argument("--limit",type=int,default=10,help="0 prints all 70 matching examples")
    args=p.parse_args()
    rows=make_rows()
    validate(rows)
    selected=choose_rows(rows,args.emotion,args.hearts,args.complexity)
    if args.limit>0: selected=selected[:args.limit]
    print(f"Showing {len(selected)} of {len(rows)} training examples")
    for row in selected:
        print("\n"+"="*65)
        print(f"{row['id']} | {row['complexity']} | emotion={row['emotion']} | heart={row['intended_heart']}")
        print("PROMPT:",row["prompt"])
        print("TARGET:",row["target"])
if __name__=="__main__":
    main()
