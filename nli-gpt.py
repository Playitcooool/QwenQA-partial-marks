import pandas as pd
import multiprocessing as mp
import requests
from tqdm import tqdm
from argparse import ArgumentParser
import os
import time
import json

from llm_tools import LLMCache

# 在这里填写您的 Qwen-Plus API 密钥
API_KEY = "sk-846a6816c1144eeea8c256c6cfc3bfb2"

prompt_template = """Please identify whether the premise entails or contradicts the hypothesis in the following premise and hypothesis. The answer should be exact "entailment", "contradiction", or "neutral". Provide only the answer from the three options. Do not provide explanations.

Premise: {premise}
Hypothesis: {hypothesis}

Is it entailment, contradiction, or neutral?"""

def call_qwen_plus_model(premise, hypothesis):
    prompt = prompt_template.format(premise=premise, hypothesis=hypothesis)
    
    # 优先使用环境变量中的 API 密钥，如果没有则使用代码中设置的
    api_key = os.getenv("QWEN_API_KEY", API_KEY)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "disable"  # 确保禁用SSE
    }
    
    payload = {
        "model": "qwen-plus",  # 使用Qwen-Plus模型
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
            "result_format": "message"  # 确保使用正确的响应格式
        }
    }
    
    try:
        # 使用正确的Qwen API端点
        response = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"API错误: {response.status_code} - {response.text}")
            return "error"
        
        result = response.json()
        
        # 检查API响应格式
        if "output" in result and "choices" in result["output"]:
            return result['output']['choices'][0]['message']['content'].strip()
        elif "output" in result and "text" in result["output"]:
            return result["output"]["text"].strip()
        else:
            print(f"未知API响应格式: {json.dumps(result, indent=2)}")
            return "error"
    
    except requests.exceptions.Timeout:
        print("API调用超时")
        return "timeout"
    except Exception as e:
        print(f"调用Qwen API时出错: {str(e)}")
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
    
    # 添加指数退避重试机制
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
                print(f"处理行 {i} 时出错 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                time.sleep(retry_delay * (2 ** attempt))  # 指数退避
            else:
                print(f"处理行 {i} 失败: {str(e)}")
                # 返回错误占位符
                placeholder = ["error"] * len(golden_statements)
                return i, placeholder, placeholder

def generate_ainf_asup(df):
    """
    基于 a2astar 和 astar2a 生成 ainf 和 asup 列
    - ainf: 1 表示系统答案被黄金答案蕴含（astar2a 含 entailment），否则 0
    - asup: 1 表示系统答案蕴含黄金答案（a2astar 含 entailment），否则 0
    """
    # 处理 ainf（系统答案被黄金答案蕴含）
    df['ainf'] = df['astar2a'].apply(
        lambda x: 1 if 'entailment' in x.lower() and 'error' not in x.lower() else 0
    )
    # 处理 asup（系统答案蕴含黄金答案）
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

    # 测试API密钥有效性
    print("测试API密钥有效性...")
    test_response = call_qwen_plus_model("test", "test")
    if test_response == "error":
        print("\n错误：API密钥无效或API调用失败")
        print("请检查以下事项：")
        print("1. 确保API密钥正确无误")
        print("2. 确保您的账户有足够的余额")
        print("3. 确保API服务已开通")
        print("4. 检查网络连接是否正常")
        print("5. 尝试在浏览器中访问 https://dashscope.console.aliyun.com/overview 查看API状态")
        exit(1)
    else:
        print(f"API测试成功，响应: {test_response[:50]}...")

    if args.dataset not in ['NQ', 'TQ']:
        raise ValueError('Invalid dataset. Only NQ or TQ allowed.')

    seed_suffix = f'-s{args.seed}' if args.seed is not None else ''
    input_file = f'data/{args.dataset}-qa2s-qwen-plus-s42.json'
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件未找到: {input_file}")
    
    df = pd.read_json(input_file)
    
    # 确保输出目录存在
    os.makedirs('data', exist_ok=True)
    
    # 备份原始数据以防处理中断
    backup_path = f'data/{args.dataset}-qa2s-qwen-plus-s42-backup.json'
    if not os.path.exists(backup_path):
        df.to_json(backup_path, orient='records', indent=2)
    
    tasks = [(i, row, args.cache_path, args.seed) for i, row in df.iterrows()]

    # 添加进度条和错误处理
    results = []
    print(f"开始处理 {len(tasks)} 条数据，使用 {args.nprocs} 个进程...")
    with mp.Pool(args.nprocs) as pool:
        # 使用imap保持顺序
        for result in tqdm(pool.imap(call_api, tasks), total=len(tasks)):
            results.append(result)
    
    # 处理结果，生成 a2astar 和 astar2a
    for i, a_to_astar, astar_to_a in results:
        df.at[i, 'a2astar'] = '||'.join(a_to_astar)
        df.at[i, 'astar2a'] = '||'.join(astar_to_a)
    
    # 生成 ainf 和 asup 列
    df = generate_ainf_asup(df)
    
    # 保存结果
    output_path = f'data/{args.dataset}-nli-qwen-plus{seed_suffix}.json'
    df.to_json(output_path, orient='records', indent=2)
    print(f"处理完成，结果已保存至: {output_path}")
    print(f"成功处理: {df['a2astar'].apply(lambda x: 'error' not in x).sum()}/{len(df)} 行")
    print(f"失败行数: {df['a2astar'].apply(lambda x: 'error' in x).sum()}")
    print(f"ainf 统计: {df['ainf'].value_counts().to_dict()} (1: 系统答案被黄金答案蕴含，0: 否)")
    print(f"asup 统计: {df['asup'].value_counts().to_dict()} (1: 系统答案蕴含黄金答案，0: 否)")