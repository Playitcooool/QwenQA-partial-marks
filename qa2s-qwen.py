import pandas as pd
import multiprocessing as mp
import requests
from tqdm import tqdm
import os
import json
import backoff
import time
from argparse import ArgumentParser
import httpx
import sys
import sqlite3

from llm_tools import LLMCache

# Fill in your Qwen-Plus API key here
API_KEY = "sk-846a6816c1144eeea8c256c6cfc3bfb2"

prompt = """Convert a question answer pair to a declarative statement, following these two examples:
Q: where is the tv show the curse of oak island filmed
A: Oak Island
S: The TV show the Curse of Oak Island is filmed on Oak Island.

Q: who wrote the first declaration of human rights
A: Cyrus
S: Cyrus wrote the first declaration of human rights

Do not provide explanations. Provide the statement only. Follow the above examples and convert this pair:
Q: {question}
A: {answer}
S:"""

def get_qwen_client():
    """Create and return Qwen API client"""
    api_key = os.getenv("QWEN_API_KEY", API_KEY)
    
    # Ensure API key is valid
    if not api_key or api_key == "your_api_key_here":
        raise ValueError("Please set a valid Qwen-Plus API key")
    
    return httpx.Client(
        base_url="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-SSE": "disable"
        },
        timeout=60  # Increase timeout duration
    )

@backoff.on_exception(backoff.expo, (httpx.RequestError, httpx.HTTPStatusError), max_time=300)
def call_qwen_backoff(question, answer, seed):
    """Call Qwen API and return response"""
    client = get_qwen_client()
    
    content = prompt.format(question=question, answer=answer)
    
    payload = {
        "model": "qwen-plus",
        "input": {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content},
            ]
        },
        "parameters": {
            "temperature": 0.0,
            "max_tokens": 300,
            "seed": seed,
            "top_p": 0.8,
            "result_format": "message"
        }
    }
    
    try:
        response = client.post("", json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Check API response format
        if "output" in result and "choices" in result["output"]:
            return result['output']['choices'][0]['message']['content'].strip()
        elif "output" in result and "text" in result["output"]:
            return result["output"]["text"].strip()
        else:
            print(f"Unknown API response format: {json.dumps(result, indent=2)}")
            return "error"
    
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            print("API rate limit exceeded, waiting to retry...")
            time.sleep(30)  # Wait 30 seconds before retrying
            raise
        print(f"API error: {e.response.status_code} - {e.response.text}")
        return "error"
    finally:
        client.close()

def call_qwen_withcache(cache, question, answer, seed):
    key = f'S{seed}--{str(question).strip()}---->{str(answer).strip()}'
    statement = cache.get(key)
    if statement is None:
        statement = call_qwen_backoff(question, answer, seed)
        cache.set(key, statement)
    return statement

def call_api(args):
    """Function to process single row of data"""
    i, row, seed = args
    
    try:
        # Ensure cache directory exists
        os.makedirs('cache', exist_ok=True)
        
        # Create cache object - no longer using closing context manager
        cache = LLMCache('cache/qa2s_cache.sqlite')
        
        golden_answers = row['golden_answer'].split('||')
        golden_statements = []
        for ans in golden_answers:
            golden_statement = call_qwen_withcache(cache, row['question'], ans, seed)
            golden_statements.append(golden_statement)
        golden_statements = '||'.join(golden_statements) 

        system_statement = call_qwen_withcache(cache, row['question'], row['system_answer'], seed)

        # Manually close cache connection
        if hasattr(cache, 'close'):
            cache.close()
        elif hasattr(cache, 'conn') and hasattr(cache.conn, 'close'):
            cache.conn.close()

        return row, golden_statements, system_statement
    except sqlite3.OperationalError as e:
        print(f"Database error (row {i}): {str(e)}")
        return row, "error", "error"
    except Exception as e:
        print(f"Error processing row {i}: {str(e)}")
        # Return error placeholders
        return row, "error", "error"

def run_by_dataset(dataset, args):
    input_file = f'data/{dataset}-reformatted.jsonl'
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    try:
        df = pd.read_json(input_file, lines=True)
    except Exception as e:
        print(f"Failed to read file: {str(e)}")
        # Try other format
        try:
            df = pd.read_json(input_file)
            print("Successfully read file in non-line format")
        except:
            raise FileNotFoundError(f"Unable to read file: {input_file}")

    if args.samplesize:
        df = df.sample(args.samplesize, random_state=42)

    print(f'Processing {df.shape[0]} rows of data...')

    # Create output directory
    os.makedirs('data', exist_ok=True)
    os.makedirs('cache', exist_ok=True)  # Ensure cache directory exists
    
    # Backup original data
    backup_path = f'data/{dataset}-reformatted-backup.jsonl'
    if not os.path.exists(backup_path):
        try:
            df.to_json(backup_path, orient='records', lines=True)
            print(f"Created data backup: {backup_path}")
        except Exception as e:
            print(f"Failed to create backup: {str(e)}")
    
    # Prepare tasks
    tasks = [(i, row, args.seed) for i, row in df.iterrows()]
    
    # Add progress bar and error handling
    results = []
    print(f"Starting processing of {len(tasks)} items, using {args.nprocs} processes...")
    
    if args.nprocs == 1:
        print("Using single-process mode...")
        for task in tqdm(tasks, desc="Processing progress"):
            results.append(call_api(task))
    else:
        print(f"Using multi-process mode ({args.nprocs} processes)...")
        # Use imap_unordered to avoid ordering issues
        with mp.Pool(args.nprocs) as pool:
            for result in tqdm(pool.imap_unordered(call_api, tasks), total=len(tasks), desc="Processing progress"):
                results.append(result)
    
    # Process results
    print("Processing results...")
    # Since using imap_unordered, results may be out of order, need to sort by index
    results.sort(key=lambda x: x[0].name if hasattr(x[0], 'name') else x[0][0])
    
    for row, golden_statements, system_statement in results:
        idx = row.name if hasattr(row, 'name') else row[0]
        df.at[idx, 'golden_statement'] = golden_statements
        df.at[idx, 'system_statement'] = system_statement
    
    # Save results
    output_path = f'data/{dataset}-qa2s-qwen-plus-s{args.seed}.json'
    try:
        df.to_json(output_path, orient='records', indent=2)
        print(f"Processing completed, results saved to: {output_path}")
    except Exception as e:
        print(f"Failed to save results: {str(e)}")
        # Try alternative format
        try:
            df.to_json(output_path, orient='records')
            print("Successfully saved results in line format")
        except:
            print("Unable to save results file")
    
    # Count success/failure cases
    if 'golden_statement' in df.columns:
        success_count = df['golden_statement'].apply(lambda x: 'error' not in x).sum()
        print(f"Successfully processed: {success_count}/{len(df)} rows")
        print(f"Failed rows: {len(df) - success_count}")
    else:
        print("Unable to count success/failure cases")

def run_nq(args):
    run_by_dataset('NQ', args)

def run_tq(args):
    run_by_dataset('TQ', args)

def test_api_key():
    """Test API key validity"""
    print("Testing API key validity...")
    try:
        test_response = call_qwen_backoff("test", "test", 42)
        if test_response == "error":
            raise ValueError("API call failed")
        print(f"API test successful, response: {test_response[:50]}...")
        return True
    except Exception as e:
        print(f"API test failed: {str(e)}")
        print("Please check the following:")
        print("1. Ensure the API key is correct")
        print("2. Ensure your account has sufficient balance")
        print("3. Ensure the API service is activated")
        print("4. Check if network connection is normal")
        return False

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='', choices=['NQ', 'TQ'], help='Dataset name (NQ or TQ)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--nprocs', type=int, default=8, help='Number of processes')
    parser.add_argument('--samplesize', type=int, help="Optional number of samples to process")
    parser.add_argument('--api_key', type=str, help="Optional API key to override default in code")
    args = parser.parse_args()

    # Test API key validity
    if not test_api_key():
        sys.exit(1)

    if args.dataset == 'NQ':
        run_nq(args)
    elif args.dataset == 'TQ':
        run_tq(args)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')