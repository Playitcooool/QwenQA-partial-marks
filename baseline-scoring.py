
import json
import multiprocessing as mp
import os
import pandas as pd
from tqdm import tqdm
import backoff

# Use dashscope (Alibaba Cloud LLM service)
from dashscope import Generation
from llm_tools import LLMCache

# Set Qwen-Plus API Key
import os
os.environ['DASHSCOPE_API_KEY'] = 'sk-846a6816c1144eeea8c256c6cfc3bfb2'  # Replace with your API Key

# Scoring prompt (same as original code)
prompt = (
    "Here is a question, a set of golden answers "
    "(split with /), an AI-generated answer. "
    "Can you judge whether the AI-generated answer is correct according to the question and golden answers? Simply give a score from 1 to 5.\n"
    "1: The AI-generated answer is completely wrong.\n"
    "2: The AI-generated answer is mostly wrong.\n"
    "3: The AI-generated answer is neither wrong nor right.\n"
    "4: The AI-generated answer is mostly right.\n"
    "5: The AI-generated answer is completely right.\n"
    "\n"
    "Question: {question}\n"
    "Golden answers: {golden_answer}\n"
    "AI answer: {system_answer}\n"
)



@backoff.on_exception(backoff.expo, Exception, max_time=600)  # Qwen may need longer retry time
def call_qwen_backoff(question, golden_answer, system_answer):
    """Call Qwen-Plus API for scoring"""
    response = Generation.call(
        model="qwen-plus",
        prompt=prompt.format(question=question, golden_answer=golden_answer, system_answer=system_answer),
        temperature=0.0,  # Lower randomness for stable results
        max_tokens=10,    # Only need a number 1-5, not long
        seed=42,          # Fixed random seed for reproducibility
    )
    if response.output is None or getattr(response.output, "text", None) is None:
        return "-1"  # Or return a default value
    return response.output.text.strip()  # Extract score number


def call_qwen_withcache(cache, question, golden_answer, system_answer):
    """Qwen-Plus call with cache"""
    key = f'{str(question).strip()}---->{str(golden_answer).strip()}---->{str(system_answer).strip()}'
    score = cache.get(key)
    if score is None:
        score = call_qwen_backoff(question, golden_answer, system_answer)
        cache.set(key, score)
    return score
def call_qwen(args):
    """Scoring function for a single data row"""
    cache = LLMCache('cache/baseline_score.sqlite')  # Use shared memory cache (faster)
    row = args[1]
    if row['a2astar'] == row['astar2a']:
        return row, -1  # Special case handling
    score = call_qwen_withcache(cache, row['question'], row['golden_answer'], row['system_answer'])
    return row, score



def run_by_dataset(dataset):
    """Process the specified dataset"""
    df = pd.read_json(f'data/{dataset}-nli-gpt35.json')
    print(f'Computing {df.shape[0]} rows...')

    inputs = list(df.iterrows())  # Convert to list to avoid iterator issues

    with mp.Pool(4) as pool, open(f'cache/_tmp_{dataset}_baselinescore.jsonl', 'w') as f:  # Reduce number of processes
        for row, score in tqdm(
            pool.imap_unordered(call_qwen, inputs, chunksize=16),  # Increase chunksize to reduce communication overhead
            total=df.shape[0],
            desc=f"Scoring {dataset}"
        ):
            row['baseline_score'] = score
            f.write(json.dumps(row.to_dict()) + '\n')
    
    # Merge results to final file
    with open(f'cache/_tmp_{dataset}_baselinescore.jsonl', 'r') as f:
        df_result = pd.read_json(f, lines=True)
        if 'id' in df_result.columns:
            df_result = df_result.drop(columns=['id'])  # Remove possible id column
        df_result.to_json(f'data/{dataset}-baselinescore-qwen-plus.json', orient='records', indent=2)



def run_nq():
    """Run NQ dataset"""
    run_by_dataset('NQ')

def run_tq():
    """Run TQ dataset"""
    run_by_dataset('TQ')

if __name__ == '__main__':
    run_tq()  # Default to running TQ dataset