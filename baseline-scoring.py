import json
import multiprocessing as mp
import os
import pandas as pd
from tqdm import tqdm
import backoff

# 使用 dashscope（阿里云大模型服务）
from dashscope import Generation
from llm_tools import LLMCache

# 配置 Qwen-Plus 的 API Key
import os
os.environ['DASHSCOPE_API_KEY'] = 'sk-846a6816c1144eeea8c256c6cfc3bfb2'  # 替换为你的 API Key

# 评分 prompt（与原代码相同）
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


@backoff.on_exception(backoff.expo, Exception, max_time=600)  # Qwen 可能需要更长的重试时间
def call_qwen_backoff(question, golden_answer, system_answer):
    """调用 Qwen-Plus API 进行评分"""
    response = Generation.call(
        model="qwen-plus",
        prompt=prompt.format(question=question, golden_answer=golden_answer, system_answer=system_answer),
        temperature=0.0,  # 降低随机性，确保结果稳定
        max_tokens=10,    # 评分只需 1-5 的数字，不需要太长
        seed=42,          # 固定随机种子，确保可复现性
    )
    if response.output is None or getattr(response.output, "text", None) is None:
        return "-1"  # 或者返回一个默认值
    return response.output.text.strip()  # 提取评分数字


def call_qwen_withcache(cache, question, golden_answer, system_answer):
    """带缓存的 Qwen-Plus 调用"""
    key = f'{str(question).strip()}---->{str(golden_answer).strip()}---->{str(system_answer).strip()}'
    score = cache.get(key)
    if score is None:
        score = call_qwen_backoff(question, golden_answer, system_answer)
        cache.set(key, score)
    return score
def call_qwen(args):
    """单条数据的评分函数"""
    cache = LLMCache('cache/baseline_score.sqlite')  # 使用共享内存缓存（更快）
    row = args[1]
    if row['a2astar'] == row['astar2a']:
        return row, -1  # 特殊情况处理
    score = call_qwen_withcache(cache, row['question'], row['golden_answer'], row['system_answer'])
    return row, score


def run_by_dataset(dataset):
    """处理指定数据集"""
    df = pd.read_json(f'data/{dataset}-nli-gpt35.json')
    print(f'Computing {df.shape[0]} rows...')

    inputs = list(df.iterrows())  # 转换为列表，避免迭代器问题

    with mp.Pool(4) as pool, open(f'cache/_tmp_{dataset}_baselinescore.jsonl', 'w') as f:  # 减少进程数
        for row, score in tqdm(
            pool.imap_unordered(call_qwen, inputs, chunksize=16),  # 增大 chunksize 减少通信开销
            total=df.shape[0],
            desc=f"Scoring {dataset}"
        ):
            row['baseline_score'] = score
            f.write(json.dumps(row.to_dict()) + '\n')
    
    # 合并结果到最终文件
    with open(f'cache/_tmp_{dataset}_baselinescore.jsonl', 'r') as f:
        df_result = pd.read_json(f, lines=True)
        if 'id' in df_result.columns:
            df_result = df_result.drop(columns=['id'])  # 移除可能的 id 列
        df_result.to_json(f'data/{dataset}-baselinescore-qwen-plus.json', orient='records', indent=2)


def run_nq():
    """运行 NQ 数据集"""
    run_by_dataset('NQ')


def run_tq():
    """运行 TQ 数据集"""
    run_by_dataset('TQ')


if __name__ == '__main__':
    run_tq()  # 默认运行 NQ 数据集