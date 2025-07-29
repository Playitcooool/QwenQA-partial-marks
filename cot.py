import json
import multiprocessing as mp
import time
from functools import partial

# Use dashscope (Alibaba Cloud large model service)
from dashscope import Generation
import backoff
import pandas as pd
from tqdm import tqdm

from llm_tools import LLMCache

# Configure Qwen-Plus API Key
import os
os.environ['DASHSCOPE_API_KEY'] = 'sk-846a6816c1144eeea8c256c6cfc3bfb2'  # Replace with your API Key

# Qwen-Plus prompt
prompt = (
    "We have two statements S1 (the premise) and S2 (the hypothesis). S1 entails S2.\n"
    "\n"
    "S1: {s1}\n\n"
    "S2: {s2}\n\n"
    "Now, list the reasoning process step by step to show how S2 can be deduced from S1.\n"
    "List the steps as numbered statements, starting from 1.\n"
    "If a step involves information not mentioned in S1 and S2, append [[INFO]] after the step.\n"
    "If an assumption must be made to deduce S2 from S1, append [[ASSUMPTION]] after the step.\n"
    "Provide the reasoning steps only. Do not include any other information.\n"
)


@backoff.on_exception(backoff.expo, Exception, max_time=600)  # Increase maximum retry time
def call_qwen_backoff(s1, s2):
    """Call Qwen-Plus API to generate reasoning process"""
    response = Generation.call(
        model="qwen-plus",
        prompt=prompt.format(s1=s1, s2=s2),
        temperature=0.0,
        max_tokens=300,
        seed=42,
    )
    return response.output.text.strip()  # 确保返回的文本无多余空格


@backoff.on_exception(backoff.expo, Exception, max_time=600)
def call_qwen_score(s1, s2, chain):
    """Call Qwen-Plus API to score reasoning difficulty"""
    score_prompt = (
        "Based on the reasoning steps, rate how hard it is to deduce S2 from S1.\n"
        "1: Very easy\n"
        "2: Easy\n"
        "3: Neither easy nor hard\n"
        "4: Hard\n"
        "5: Very hard\n"
        "Return a number from 1 to 5 only.\n"
    )
    full_prompt = f"{prompt.format(s1=s1, s2=s2)}\n\n{chain}\n\n{score_prompt}"
    response = Generation.call(
        model="qwen-plus",
        prompt=full_prompt,
        temperature=0.0,
        max_tokens=10,  # 评分只需 1-5 的数字
        seed=42,
    )
    return response.output.text.strip()


def cached_call_score_qwen(cache: LLMCache, s1, s2, chain):
    """Score call with cache"""
    key = f"{s1}---->{s2}"
    result = cache.get(key)
    if result is None:
        result = call_qwen_score(s1, s2, chain)
        cache.set(key, result)
    return result


def cached_call_cot_qwen(cache: LLMCache, s1, s2):
    """CoT call with cache"""
    key = f"{s1}---->{s2}"
    result = cache.get(key)
    if result is None:
        result = call_qwen_backoff(s1, s2)
        cache.set(key, result)
    return result


def cot_single_gpt35(args):
    """Single data CoT generation (with error handling)"""
    i, row, seed = args
    try:
        golden_statements = row["golden_statement"].split("||")
        cache = LLMCache("cache/cot-cache-qwen.sqlite")

        direction = "astar2a" if (row["ainf"] == 1 and row["asup"] == 0) else "a2astar"
        which_entails = ["entailment" in x.lower() for x in row[direction].split("||")]

        chains = []
        for gs, entails in zip(golden_statements, which_entails):
            if not entails:
                continue
            s1 = gs if direction == "astar2a" else row["system_statement"]
            s2 = row["system_statement"] if direction == "astar2a" else gs
            chains.append(cached_call_cot_qwen(cache, s1, s2))
        return i, row, chains
    except Exception as e:
        print(f"Error processing row {i}: {str(e)}")
        return i, row, []  # Return empty list to indicate failure


def score_single_gpt35(args):
    """Single data scoring (with error handling)"""
    i, row, seed = args
    try:
        golden_statements = row["golden_statement"].split("||")
        cache = LLMCache("cache/cot-cache-qwen.sqlite")
        score_cache = LLMCache("cache/cot-score-qwen.sqlite")

        direction = "astar2a" if (row["ainf"] == 1 and row["asup"] == 0) else "a2astar"
        which_entails = ["entailment" in x.lower() for x in row[direction].split("||")]

        scores = []
        for gs, entails in zip(golden_statements, which_entails):
            if not entails:
                continue
            s1 = gs if direction == "astar2a" else row["system_statement"]
            s2 = row["system_statement"] if direction == "astar2a" else gs
            chain = cached_call_cot_qwen(cache, s1, s2)
            score = cached_call_score_qwen(score_cache, s1, s2, chain)
            scores.append(score)
        return i, row, scores
    except Exception as e:
        print(f"Error scoring row {i}: {str(e)}")
        return i, row, []  # Return empty list to indicate failure


def run_cot_qwen():
    """Run CoT generation (optimized version)"""
    df = pd.read_json("data/NQ-nli-qwen-plus-s42.json")
    inputs = [(i, row, 42) for i, row in df.iterrows()]  # Fixed seed for reproducibility

    # Use single process to test for deadlock (uncomment for debugging)
    # for i, row, chains in tqdm(map(cot_single_gpt35, inputs), total=len(inputs)):
    #     pass

    # Multiprocessing mode (limit process count to reduce contention)
    with mp.Pool(4) as pool, open("data/cot-qwen.jsonl", "w") as f:  # Reduced from 8 to 4
        for i, row, chains in tqdm(
            pool.imap_unordered(cot_single_gpt35, inputs), 
            total=len(inputs),
            desc="Generating CoT"
        ):
            output = dict(row)  # Copy original data
            output["chains"] = chains
            f.write(json.dumps(output) + "\n")


def run_score_qwen():
    """Run scoring (optimized version)"""
    df = pd.read_json("data/NQ-nli-qwen-plus-s42.json")
    inputs = [(i, row, 42) for i, row in df.iterrows()]

    with mp.Pool(4) as pool, open("data/cot-score-qwen.jsonl", "w") as f:
        for i, row, scores in tqdm(
            pool.imap_unordered(score_single_gpt35, inputs),
            total=len(inputs),
            desc="Scoring CoT"
        ):
            output = dict(row)
            output["scores"] = scores
            f.write(json.dumps(output) + "\n")


if __name__ == "__main__":

    # Run main program
    print("Running CoT generation...")
    run_cot_qwen()
    print("Running scoring...")
    run_score_qwen()
    print("Done!")