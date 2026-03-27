import pandas as pd
from openai import OpenAI
import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# 配置日志
LOG_FILE = "logs/sentiment_analysis.log"
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局token计数器
total_tokens = 0
total_requests = 0

# 读取配置
def load_config():
    config = {}
    if os.path.exists('config.py'):
        import config
        config['api_key'] = getattr(config, 'API_KEY', os.environ.get('DEEPSEEK_API_KEY'))
        config['base_url'] = getattr(config, 'BASE_URL', 'https://api.deepseek.com')
    else:
        config['api_key'] = os.environ.get('DEEPSEEK_API_KEY')
        config['base_url'] = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')

    if not config['api_key']:
        raise ValueError("请在 config.py 中设置 API_KEY 或设置环境变量 DEEPSEEK_API_KEY")
    return config

config = load_config()

# 配置 OpenAI 客户端 (DeepSeek API)
client = OpenAI(
    api_key=config['api_key'],
    base_url=config['base_url']
)

# 配置多个数据文件
DATA_FILES = [
    {
        "input": "data/douyin_cleaned.xlsx",
        "output": "results/douyin_sentiment_result.csv",
        "comment_cols": ["清洗后纯评论（无地名/无新增词）"]
    },
    {
        "input": "data/weibo_comments_cleaned.xlsx",
        "output": "results/weibo_sentiment_result.csv",
        "comment_cols": ["cleaned_content"]
    },
    {
        "input": "data/xiaohongshu_comments_cleaned.xlsx",
        "output": "results/xiaohongshu_sentiment_result.csv",
        "comment_cols": ["cleaned_content"]
    }
]

# 定义函数对单条评论进行情感分析
def classify_comment(comment):
    global total_tokens, total_requests
    try:
        messages = [
            {"role": "system", "content": """
                你是一个情感分析助手。请根据以下示例判断新文本的情感倾向，并给出简要解析。

                示例格式：
                评论：好可爱的四位男主，还有我组做大做强！
                情感：正面
                解析：该评论使用了"好可爱"、"做大做强"等积极词汇，表达了对内容和团队的赞赏与支持，整体情感倾向明显为正面。

                评论：真好，不是无声无息的联动。绣球上去好漂亮，梦回一些古早古装剧，想拥有。
                情感：正面
                解析：评论中使用了"真好"、"好漂亮"、"想拥有"等积极词汇，表达了对联动内容的喜爱和期待，整体情感倾向为正面。

                评论：为了国际化去中国元素，再为了流水搞个联动哄哄国内玩家，属实把两面派当明白了
                情感：负面
                解析：该评论批评了公司为国际化而移除中国元素，又通过联动讨好国内玩家的行为，指责其"两面派"做法，带有讽刺和不满情绪，整体情感倾向为负面。

                评论：依旧和我无关（
                情感：中性
                解析：该评论只是表达与自身无关的态度，没有明显的情感倾向，属于中性评论。

                评论：谈恋爱＝活受罪
                情感：负面
                解析："活受罪"是网络用语，表示活该受罪的意思，带有明显的负面情绪，表达了对谈恋爱的抗拒和不满。

                请严格按照以下格式输出，不要添加任何额外内容：
                情感：[正面/负面/中性]
                解析：[简要分析理由，50-100字左右]
                """},
            {"role": "user", "content": f"请分析以下评论的情感倾向：[comment begin]{comment} [comment end]"}
        ]
        completion = client.chat.completions.create(
            model="deepseek-chat",  # DeepSeek 模型
            messages=messages
        )
        # 统计token使用量
        if hasattr(completion, 'usage') and completion.usage:
            tokens = completion.usage.total_tokens
            total_tokens += tokens
            total_requests += 1
            if total_requests % 50 == 0:
                logger.info(f"[Token统计] 请求次数: {total_requests}, 总token: {total_tokens}")

        # 提取返回内容中的情感和解析
        content = completion.choices[0].message.content

        # 提取情感标签 - 匹配完整词汇
        sentiment_match = re.search(r'情感[：:]\s*(正面|负面|中性)', content)
        sentiment = sentiment_match.group(1) if sentiment_match else "无法分类"

        # 提取解析内容 - 修复正则，让解析更灵活
        lines = content.split('\n')
        analysis = ""
        for line in lines:
            if line.strip().startswith('解析'):
                analysis = line.split('解析')[1].strip('：:').strip()
                break
        if not analysis:
            # 备用方法：提取"解析："之后的内容
            match = re.search(r'解析[：:]\s*(.+)', content)
            analysis = match.group(1).strip() if match else "无法解析"

        # 映射情感词到数字
        sentiment_map = {"正面": 1, "中性": 0, "负面": -1}
        sentiment_num = sentiment_map.get(sentiment, "无法分类")

        return sentiment_num, analysis

    except Exception as e:
        logger.error(f"API调用失败: {str(e)}")
        return "无法分类", "无法解析"

# 查找评论列
def find_comment_column(df, comment_cols):
    for col in comment_cols:
        if col in df.columns:
            return col
    return None

# 读取文件（自动处理编码）
def read_data_file(file_config):
    input_path = file_config["input"]
    comment_cols = file_config["comment_cols"]

    # 根据文件扩展名选择读取方式
    if input_path.endswith('.csv'):
        # 尝试多种编码
        for encoding in ['utf-8', 'gb18030', 'gbk', 'utf-8-sig']:
            try:
                df = pd.read_csv(input_path, encoding=encoding)
                col = find_comment_column(df, comment_cols)
                if col:
                    logger.info(f"成功读取 CSV 文件，使用编码: {encoding}, 评论列: {col}")
                    return df, col
            except UnicodeDecodeError:
                continue
        logger.error(f"无法读取 CSV 文件: {input_path}")
        return None, None
    else:
        # Excel 文件
        try:
            df = pd.read_excel(input_path)
            col = find_comment_column(df, comment_cols)
            if col:
                logger.info(f"成功读取 Excel 文件, 评论列: {col}")
                return df, col
            else:
                logger.warning(f"未找到评论列，可用列: {list(df.columns)}")
                return None, None
        except Exception as e:
            logger.error(f"读取 Excel 文件时发生错误: {str(e)}")
            return None, None

# 处理单个数据文件
def process_file(file_config):
    input_path = file_config["input"]
    output_path = file_config["output"]
    comment_cols = file_config["comment_cols"]

    logger.info(f"开始处理: {input_path}")
    logger.info(f"输出文件: {output_path}")

    # 读取数据
    df, comment_col = read_data_file(file_config)
    if df is None or comment_col is None:
        logger.warning(f"跳过文件: {input_path}")
        return

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 检查已有结果文件，获取已处理数量
    processed_count = 0
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path, encoding='utf-8-sig')
        processed_count = len(existing_df)
        logger.info(f"已有 {processed_count} 条已处理，继续...")

    # 获取评论数据
    comments = df[comment_col].fillna('').astype(str)
    total = len(comments)
    logger.info(f"共 {total} 条评论，待处理: {total - processed_count} 条")

    # 需要处理的评论索引
    pending_indices = [i for i in range(total) if i >= processed_count]

    if len(pending_indices) == 0:
        logger.info("所有评论已处理完成")
        return

    # 并发处理配置
    MAX_WORKERS = 15  # 并发数
    BATCH_SIZE = 100  # 每处理100条保存一次

    # 使用线程池并发处理
    completed = 0
    batch_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_idx = {executor.submit(classify_comment, comments.iloc[i]): i for i in pending_indices}

        # 收集结果
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
            except Exception as e:
                result = ("无法分类", "无法解析")
                logger.error(f"处理评论索引 {idx} 时出错: {str(e)}")

            # 获取原始数据行
            row_data = df.iloc[idx].to_dict()
            # 处理返回的元组 (sentiment, analysis)
            if isinstance(result, tuple):
                row_data['情感分析结果'] = result[0]
                row_data['解析'] = result[1]
            else:
                row_data['情感分析结果'] = result
                row_data['解析'] = "无法解析"
            batch_results.append(row_data)

            completed += 1

            # 批量保存结果
            if completed % BATCH_SIZE == 0 or completed == len(pending_indices):
                # 追加写入CSV
                if os.path.exists(output_path):
                    batch_df = pd.DataFrame(batch_results)
                    batch_df.to_csv(output_path, index=False, mode='a', header=False, encoding='utf-8-sig')
                else:
                    batch_df = pd.DataFrame(batch_results)
                    batch_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                logger.info(f"进度: {completed}/{len(pending_indices)} (已保存)")
                batch_results = []  # 清空批次缓冲区

    logger.info(f"情感分析完成: {output_path}")

# 打印最终Token统计
def print_token_stats():
    global total_tokens, total_requests
    logger.info("=" * 50)
    logger.info("Token使用统计:")
    logger.info(f"  总请求次数: {total_requests}")
    logger.info(f"  总token消耗: {total_tokens}")
    # 估算费用 (DeepSeek Chat 价格: ¥1/百万tokens)
    estimated_cost = total_tokens / 1_000_000 * 1
    logger.info(f"  预估费用: ¥{estimated_cost:.4f}")
    logger.info("=" * 50)

# 主函数 - 批量处理所有数据文件
def main():
    global total_tokens, total_requests
    logger.info("=" * 50)
    logger.info("开始批量处理情感分析...")
    logger.info(f"共 {len(DATA_FILES)} 个文件需要处理")

    for i, file_config in enumerate(DATA_FILES, 1):
        logger.info(f"[{i}/{len(DATA_FILES)}] 正在处理...")
        process_file(file_config)

    logger.info("所有文件处理完成！")
    logger.info("=" * 50)

    # 打印Token统计
    print_token_stats()

if __name__ == "__main__":
    main()