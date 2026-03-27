import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据
platforms = ['抖音', '微博', '小红书']
total = [1021, 717, 1625]
positive = [492, 403, 988]
neutral = [426, 192, 536]
negative = [103, 122, 101]

# 计算占比
positive_ratio = [p/t*100 for p, t in zip(positive, total)]
negative_ratio = [n/t*100 for n, t in zip(negative, total)]
neutral_ratio = [n/t*100 for n, t in zip(neutral, total)]

# 负/正比例
neg_pos_ratio = [negative[i]/positive[i] for i in range(3)]

x = np.arange(len(platforms))
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 图1：正负评论占比
ax1 = axes[0]
bars1 = ax1.bar(x - width/2, positive_ratio, width, label='正面 (1)', color='#4CAF50')
bars2 = ax1.bar(x + width/2, negative_ratio, width, label='负面 (-1)', color='#F44336')
ax1.set_xlabel('平台', fontsize=12)
ax1.set_ylabel('占比 (%)', fontsize=12)
ax1.set_title('各平台正负评论占比', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(platforms)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar, val in zip(bars1, positive_ratio):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, negative_ratio):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

# 图2：负/正比例
ax2 = axes[1]
colors = ['#FF9800' if r > 0.2 else '#2196F3' for r in neg_pos_ratio]
bars3 = ax2.bar(x, neg_pos_ratio, width=0.5, color=colors)
ax2.set_xlabel('平台', fontsize=12)
ax2.set_ylabel('负/正比例', fontsize=12)
ax2.set_title('负面评论/正面评论 比例', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(platforms)
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='警戒线(0.2)')

# 添加数值标签
for bar, val in zip(bars3, neg_pos_ratio):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('results/sentiment_analysis_chart.png', dpi=150, bbox_inches='tight')
plt.close()

print("图表已保存到: results/sentiment_analysis_chart.png")

# 打印统计信息
print("\n=== 情感分析统计 ===")
print(f"{'平台':<8} {'总数':<6} {'正面':<6} {'中性':<6} {'负面':<6} {'正面占比':<10} {'负面占比':<10} {'负/正比':<8}")
print("-" * 70)
for i, name in enumerate(platforms):
    print(f"{name:<8} {total[i]:<6} {positive[i]:<6} {neutral[i]:<6} {negative[i]:<6} "
          f"{positive_ratio[i]:>8.1f}% {negative_ratio[i]:>8.1f}% {neg_pos_ratio[i]:>8.2f}")
