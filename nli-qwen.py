import pandas as pd
import multiprocessing as mp
import requests
from tqdm import tqdm
from argparse import ArgumentParser
import os
import time
import json

from llm_tools import LLMCache

# Fill in your Qwen-Plus API key here
API_KEY = "sk-846a6816c1144eeea8c256c6cfc3bfb2"

prompt_template = """Please identify whether the premise entails or contradicts the hypothesis in the following premise and hypothesis. The answer should be exact "entailment", "contradiction", or "neutral". Provide only the answer from the three options. Do not provide explanations.

Premise: {premise}
Hypothesis: {hypothesis}

Is it entailment, contradiction, or neutral?"""

def call_qwen_plus_model(premise, hypothesis):
    prompt = prompt_template.format(premise=premise, hypothesis=hypothesis)
    
    # Prefer API key from environment variable, fallback to hardcoded value
    api_key = os.getenv("QWEN_API_KEY", API_KEY)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "disable"  # 确保禁用SSE
    }
    
    payload = {
        "model": "qwen-plus",  # Use Qwen-Plus model
        "input": {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        },
        "parameters": {
            "temperature": 0.0,
            "max_tokens": 50,
            "top_p": 0.8,
            "result_format": "message"  # Ensure correct response format
        }
    }
    
    try:
        # Use correct Qwen API endpoint
        response = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"API error: {response.status_code} - {response.text}")
            return "error"
        
        result = response.json()
        
        # Check API response format
        if "output" in result and "choices" in result["output"]:
            return result['output']['choices'][0]['message']['content'].strip()
        elif "output" in result and "text" in result["output"]:
            return result["output"]["text"].strip()
        else:
            print(f"Unknown API response format: {json.dumps(result, indent=2)}")
            return "error"
    
    except requests.exceptions.Timeout:
        print("API call timed out")
        return "timeout"
    except Exception as e:
        print(f"Error calling Qwen API: {str(e)}")
        return "error"

def check_cache(cache_path, seed, premise, hypothesis):
    cache = LLMCache(cache_path)
    key = f"s{seed}||{premise}-->{hypothesis}"
    value = cache.get(key)
    if value is None:
        value = call_qwen_plus_model(premise, hypothesis)
        cache.set(key, value)
    return value

def call_api(args_tuple):
    i, row, cache_path, seed = args_tuple
    golden_statements = row['golden_statement'].split('||')
    
    # Add exponential backoff retry mechanism
    max_retries = 3
    retry_delay = 2  # 初始延迟秒数
    
    for attempt in range(max_retries):
        try:
            astar_to_a = [
                check_cache(cache_path, seed, ans, row['system_statement']) for ans in golden_statements
            ]
            a_to_astar = [
                check_cache(cache_path, seed, row['system_statement'], ans) for ans in golden_statements
            ]
            return i, a_to_astar, astar_to_a
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error processing row {i} (attempt {attempt+1}/{max_retries}): {str(e)}")
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            else:
                print(f"Failed processing row {i}: {str(e)}")
                # Return error placeholder
                placeholder = ["error"] * len(golden_statements)
                return i, placeholder, placeholder

def generate_ainf_asup(df):
    """
    Generate ainf and asup columns based on a2astar and astar2a
    - ainf: 1 means system answer is entailed by golden answer (astar2a contains entailment), else 0
    - asup: 1 means system answer entails golden answer (a2astar contains entailment), else 0
    """
    # Process ainf (system answer is entailed by golden answer)
    df['ainf'] = df['astar2a'].apply(
        lambda x: 1 if 'entailment' in x.lower() and 'error' not in x.lower() else 0
    )
    # Process asup (system answer entails golden answer)
    df['asup'] = df['a2astar'].apply(
        lambda x: 1 if 'entailment' in x.lower() and 'error' not in x.lower() else 0
    )
    return df

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NQ')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nprocs', type=int, default=4)
    parser.add_argument('--cache_path', type=str, default='nli-gpt-cache.sqlite')
    parser.add_argument('--api_key', type=str, help="可选的API密钥，覆盖代码中的默认设置")
    args = parser.parse_args()
    API_KEY = "sk-846a6816c1144eeea8c256c6cfc3bfb2"

    # Test API key validity
    print("Testing API key validity...")
    test_response = call_qwen_plus_model("test", "test")
    if test_response == "error":
        print("\nError: API key invalid or API call failed")
        print("Please check the following:")
        print("1. Ensure API key is correct")
        print("2. Ensure your account has sufficient balance")
        print("3. Ensure API service is enabled")
        print("4. Check your network connection")
        print("5. Try visiting https://dashscope.console.aliyun.com/overview to check API status")
        exit(1)
    else:
        print(f"API test successful, response: {test_response[:50]}...")

    if args.dataset not in ['NQ', 'TQ']:
        raise ValueError('Invalid dataset. Only NQ or TQ allowed.')

    seed_suffix = f'-s{args.seed}' if args.seed is not None else ''
    input_file = f'data/{args.dataset}-qa2s-qwen-plus-s42.json'
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    df = pd.read_json(input_file)
    
    # Ensure output directory exists
    os.makedirs('data', exist_ok=True)
    
    # Backup original data in case of interruption
    backup_path = f'data/{args.dataset}-qa2s-qwen-plus-s42-backup.json'
    if not os.path.exists(backup_path):
        df.to_json(backup_path, orient='records', indent=2)
    
    tasks = [(i, row, args.cache_path, args.seed) for i, row in df.iterrows()]

    # Add progress bar and error handling
    results = []
    print(f"Starting to process {len(tasks)} data items using {args.nprocs} processes...")
    with mp.Pool(args.nprocs) as pool:
        # 使用imap保持顺序
        for result in tqdm(pool.imap(call_api, tasks), total=len(tasks)):
            results.append(result)
    
    # Process results, generate a2astar and astar2a
    for i, a_to_astar, astar_to_a in results:
        df.at[i, 'a2astar'] = '||'.join(a_to_astar)
        df.at[i, 'astar2a'] = '||'.join(astar_to_a)
    
    # Generate ainf and asup columns
    df = generate_ainf_asup(df)
    
    # Save results
    output_path = f'data/{args.dataset}-nli-qwen-plus{seed_suffix}.json'
    df.to_json(output_path, orient='records', indent=2)
    print(f"Processing complete, results saved to: {output_path}")
    print(f"Successfully processed: {df['a2astar'].apply(lambda x: 'error' not in x).sum()}/{len(df)} rows")
    print(f"Failed rows: {df['a2astar'].apply(lambda x: 'error' in x).sum()}")
    print(f"ainf stats: {df['ainf'].value_counts().to_dict()} (1: system answer entailed by golden answer, 0: not)")
    print(f"asup stats: {df['asup'].value_counts().to_dict()} (1: system answer entails golden answer, 0: not)")