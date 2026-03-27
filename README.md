# DeepSeek Sentiment Analysis

基于 DeepSeek API 的多平台情感分析系统，用于分析抖音、微博、小红书等平台的评论数据。

## 功能

- 支持多平台数据情感分析（正面/负面/中性）
- 批量处理评论数据
- 生成可视化图表
- 输出详细分析报告

## 文件结构

```
├── sentiment_analysis.py    # 主程序
├── plot_sentiment.py       # 可视化模块
├── config.py.example       # 配置示例
├── data/                   # 原始数据
├── results/                # 分析结果
└── logs/                   # 日志文件
```

## 快速开始

1. 复制配置示例文件：
   ```bash
   cp config.py.example config.py
   ```

2. 编辑 `config.py`，填入你的 DeepSeek API Key

3. 运行分析：
   ```bash
   python sentiment_analysis.py
   ```

4. 生成可视化：
   ```bash
   python plot_sentiment.py
   ```

## 依赖

- openai
- pandas
- matplotlib
- tqdm

安装依赖：
```bash
pip install openai pandas matplotlib tqdm openpyxl
```

## 数据格式

支持 CSV 和 Excel 格式的评论数据文件。