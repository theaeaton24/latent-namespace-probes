#!/usr/bin/env python3
"""Inspect EXACT Qwen text fed to tokenizer before evaluation.

Does not load model weights or generate any responses.
"""
import argparse
from transformers import AutoTokenizer
from study import encode,prompt_environment

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--message",default="I am having a difficult day and would appreciate encouragement.")
    p.add_argument("--system-condition",choices=["tutor","none","no-system"],default="no-system")
    a=p.parse_args()
    tok=AutoTokenizer.from_pretrained(a.model)
    system,fmt=prompt_environment(a.system_condition,a.model)
    text,_=encode(tok,system,a.message,prompt_format="chat" if fmt=="auto" else fmt)
    print("System condition:",a.system_condition)
    print("Rendered prompt:\n"+text)
    print("Tokenizer token count:",len(tok(text,add_special_tokens=False)["input_ids"]))
    print("Contains default persona:", "You are Qwen, created by Alibaba Cloud" in text)

if __name__=="__main__":
    main()
