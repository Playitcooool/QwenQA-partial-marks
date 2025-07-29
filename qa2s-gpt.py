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

# 在这里填写您的 Qwen-Plus API 密钥
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
    """创建并返回Qwen API客户端"""
    api_key = os.getenv("QWEN_API_KEY", API_KEY)
    
    # 确保API密钥有效
    if not api_key or api_key == "your_api_key_here":
        raise ValueError("请设置有效的Qwen-Plus API密钥")
    
    return httpx.Client(
        base_url="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-SSE": "disable"
        },
        timeout=60  # 增加超时时间
    )

@backoff.on_exception(backoff.expo, (httpx.RequestError, httpx.HTTPStatusError), max_time=300)
def call_qwen_backoff(question, answer, seed):
    """调用Qwen API并返回响应"""
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
        
        # 检查API响应格式
        if "output" in result and "choices" in result["output"]:
            return result['output']['choices'][0]['message']['content'].strip()
        elif "output" in result and "text" in result["output"]:
            return result["output"]["text"].strip()
        else:
            print(f"未知API响应格式: {json.dumps(result, indent=2)}")
            return "error"
    
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            print("API速率限制，等待重试...")
            time.sleep(30)  # 等待30秒后重试
            raise
        print(f"API错误: {e.response.status_code} - {e.response.text}")
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
    """处理单行数据的函数"""
    i, row, seed = args
    
    try:
        # 确保缓存目录存在
        os.makedirs('cache', exist_ok=True)
        
        # 创建缓存对象 - 不再使用closing上下文管理器
        cache = LLMCache('cache/qa2s_cache.sqlite')
        
        golden_answers = row['golden_answer'].split('||')
        golden_statements = []
        for ans in golden_answers:
            golden_statement = call_qwen_withcache(cache, row['question'], ans, seed)
            golden_statements.append(golden_statement)
        golden_statements = '||'.join(golden_statements) 

        system_statement = call_qwen_withcache(cache, row['question'], row['system_answer'], seed)

        # 手动关闭缓存连接
        if hasattr(cache, 'close'):
            cache.close()
        elif hasattr(cache, 'conn') and hasattr(cache.conn, 'close'):
            cache.conn.close()

        return row, golden_statements, system_statement
    except sqlite3.OperationalError as e:
        print(f"数据库错误 (行 {i}): {str(e)}")
        return row, "error", "error"
    except Exception as e:
        print(f"处理行 {i} 时出错: {str(e)}")
        # 返回错误占位符
        return row, "error", "error"

def run_by_dataset(dataset, args):
    input_file = f'data/{dataset}-reformatted.jsonl'
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件未找到: {input_file}")
    
    try:
        df = pd.read_json(input_file, lines=True)
    except Exception as e:
        print(f"读取文件失败: {str(e)}")
        # 尝试其他格式
        try:
            df = pd.read_json(input_file)
            print("成功以非行格式读取文件")
        except:
            raise FileNotFoundError(f"无法读取文件: {input_file}")

    if args.samplesize:
        df = df.sample(args.samplesize, random_state=42)

    print(f'处理 {df.shape[0]} 行数据...')

    # 创建输出目录
    os.makedirs('data', exist_ok=True)
    os.makedirs('cache', exist_ok=True)  # 确保缓存目录存在
    
    # 备份原始数据
    backup_path = f'data/{dataset}-reformatted-backup.jsonl'
    if not os.path.exists(backup_path):
        try:
            df.to_json(backup_path, orient='records', lines=True)
            print(f"创建数据备份: {backup_path}")
        except Exception as e:
            print(f"备份失败: {str(e)}")
    
    # 准备任务
    tasks = [(i, row, args.seed) for i, row in df.iterrows()]
    
    # 添加进度条和错误处理
    results = []
    print(f"开始处理 {len(tasks)} 条数据，使用 {args.nprocs} 个进程...")
    
    if args.nprocs == 1:
        print("使用单进程模式...")
        for task in tqdm(tasks, desc="处理进度"):
            results.append(call_api(task))
    else:
        print(f"使用多进程模式 ({args.nprocs} 进程)...")
        # 使用imap_unordered避免顺序问题
        with mp.Pool(args.nprocs) as pool:
            for result in tqdm(pool.imap_unordered(call_api, tasks), total=len(tasks), desc="处理进度"):
                results.append(result)
    
    # 处理结果
    print("处理结果...")
    # 由于使用imap_unordered，结果顺序可能不同，需要按索引排序
    results.sort(key=lambda x: x[0].name if hasattr(x[0], 'name') else x[0][0])
    
    for row, golden_statements, system_statement in results:
        idx = row.name if hasattr(row, 'name') else row[0]
        df.at[idx, 'golden_statement'] = golden_statements
        df.at[idx, 'system_statement'] = system_statement
    
    # 保存结果
    output_path = f'data/{dataset}-qa2s-qwen-plus-s{args.seed}.json'
    try:
        df.to_json(output_path, orient='records', indent=2)
        print(f"处理完成，结果已保存至: {output_path}")
    except Exception as e:
        print(f"保存结果失败: {str(e)}")
        # 尝试其他格式
        try:
            df.to_json(output_path, orient='records')
            print("成功以行格式保存结果")
        except:
            print("无法保存结果文件")
    
    # 统计成功/失败情况
    if 'golden_statement' in df.columns:
        success_count = df['golden_statement'].apply(lambda x: 'error' not in x).sum()
        print(f"成功处理: {success_count}/{len(df)} 行")
        print(f"失败行数: {len(df) - success_count}")
    else:
        print("无法统计成功/失败情况")

def run_nq(args):
    run_by_dataset('NQ', args)

def run_tq(args):
    run_by_dataset('TQ', args)

def test_api_key():
    """测试API密钥有效性"""
    print("测试API密钥有效性...")
    try:
        test_response = call_qwen_backoff("test", "test", 42)
        if test_response == "error":
            raise ValueError("API调用失败")
        print(f"API测试成功，响应: {test_response[:50]}...")
        return True
    except Exception as e:
        print(f"API测试失败: {str(e)}")
        print("请检查以下事项：")
        print("1. 确保API密钥正确无误")
        print("2. 确保您的账户有足够的余额")
        print("3. 确保API服务已开通")
        print("4. 检查网络连接是否正常")
        return False

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TQ', choices=['NQ', 'TQ'], help='数据集名称 (NQ 或 TQ)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--nprocs', type=int, default=8, help='进程数')
    parser.add_argument('--samplesize', type=int, help="可选的处理样本数量")
    parser.add_argument('--api_key', type=str, help="可选的API密钥，覆盖代码中的默认设置")
    args = parser.parse_args()

    # 测试API密钥有效性
    if not test_api_key():
        sys.exit(1)

    if args.dataset == 'NQ':
        run_nq(args)
    elif args.dataset == 'TQ':
        run_tq(args)
    else:
        raise ValueError(f'未知数据集: {args.dataset}')