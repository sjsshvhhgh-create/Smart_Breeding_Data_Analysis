import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# 牛只全生命周期数据分析脚本
# 运行方式：python src/data_analysis.py
# 所有图表保存到 data/figures/ 目录
# ============================================================

# 设置路径
DATA_DIR = "./data/processed"
FIG_DIR = "./figures/data_analysis"
os.makedirs(FIG_DIR, exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

# 读取数据
df = pd.read_csv(os.path.join(DATA_DIR, "牛只全生命周期分析宽表.csv"))
print(f"\n{'='*55}")
print(f"🐮 牛只全生命周期分析报告")
print(f"{'='*55}")
print(f"✅ 加载数据: {df.shape[0]} 行 × {df.shape[1]} 列\n")

# ============================================================
# 1. 数据概览
# ============================================================
print("📋 字段信息:")
for col in df.columns:
    missing = df[col].isna().sum()
    dtype = df[col].dtype
    print(f"   {col:20s}  {str(dtype):10s}  缺失: {missing:4d}")

print(f"\n📊 数值型统计摘要:")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(df[num_cols].describe().T.round(2).to_string())

# ============================================================
# 2. 分类字段分布
# ============================================================
print(f"\n{'='*55}")
print("📊 分类字段分布")
print(f"{'='*55}")

for col in ['牛只品种', '性别', '最终结局']:
    if col in df.columns:
        print(f"\n  【{col}】")
        vc = df[col].value_counts()
        for k, v in vc.items():
            print(f"    {k}: {v} 头 ({v/len(df)*100:.1f}%)")

# ============================================================
# 3. 可视化分析
# ============================================================
print(f"\n{'='*55}")
print("📈 正在生成可视化图表...")
print(f"{'='*55}")

# 3.1 体重分布
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df['初始体重'].dropna(), bins=30, kde=True, ax=axes[0], color='steelblue')
axes[0].set_title('初始体重分布')
axes[0].set_xlabel('体重(kg)')
if '最新体重' in df.columns:
    sns.histplot(df['最新体重'].dropna(), bins=30, kde=True, ax=axes[1], color='coral')
    axes[1].set_title('最新体重分布')
    axes[1].set_xlabel('体重(kg)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '01_体重分布.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  01_体重分布.png ✅")

# 3.2 日增重与转化率
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
if '平均日增重' in df.columns:
    sns.histplot(df['平均日增重'].dropna(), bins=30, kde=True, ax=axes[0], color='green')
    axes[0].set_title('平均日增重 (kg/天)')
if '精料转化率' in df.columns:
    v = df['精料转化率'].dropna()
    v = v[(v > 0) & (v < 1)]
    sns.histplot(v, bins=30, kde=True, ax=axes[1], color='purple')
    axes[1].set_title('精料转化率')
if '总饲料转化率' in df.columns:
    v2 = df['总饲料转化率'].dropna()
    v2 = v2[(v2 > 0) & (v2 < 0.5)]
    sns.histplot(v2, bins=30, kde=True, ax=axes[2], color='orange')
    axes[2].set_title('总饲料转化率')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '02_日增重与转化率.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  02_日增重与转化率.png ✅")

# 3.3 生命周期分布
if '总生命周期(天)' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df['总生命周期(天)'].dropna(), bins=40, kde=True, color='teal')
    ax.set_title('牛只生命周期分布')
    ax.set_xlabel('天数')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '03_生命周期分布.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  03_生命周期分布.png ✅")

# 3.4 相关性热力图
core_cols = ['初始体重', '平均日增重', '最新体重', '总增重',
             '累计用药次数', '累计防疫次数', '总生命周期(天)',
             '精料总量_kg', '饲料总花费_元', '喂养天数',
             '精料转化率', '总饲料转化率']
existing = [c for c in core_cols if c in df.columns]
if len(existing) > 2:
    corr = df[existing].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                vmin=-1, vmax=1, center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title('核心指标相关性热力图')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '04_相关性热力图.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  04_相关性热力图.png ✅")

# 3.5 分组对比箱线图
if '最终结局' in df.columns:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, col in enumerate(['初始体重', '平均日增重', '总生命周期(天)']):
        if col in df.columns:
            sns.boxplot(data=df, x='最终结局', y=col, ax=axes[i], palette='Set2')
            axes[i].set_title(f'{col} × 最终结局')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '05_分组对比.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  05_分组对比.png ✅")


# ============================================================
# 4. 效率排行
# ============================================================
print(f"\n{'='*55}")
print("🏆 效率排行")
print(f"{'='*55}")

if '平均日增重' in df.columns:
    top = df.nlargest(10, '平均日增重')[['资产 ID', '牛只品种', '性别', '初始体重', '平均日增重', '最终结局']]
    print("\n日增重 TOP 10:")
    print(top.to_string(index=False))

if '精料转化率' in df.columns:
    tmp = df[df['精料转化率'].between(0.01, 1)].copy()
    top_fcr = tmp.nlargest(10, '精料转化率')[['资产 ID', '牛只品种', '初始体重', '总增重', '精料总量_kg', '精料转化率']]
    print("\n精料转化率 TOP 10:")
    print(top_fcr.to_string(index=False))

# ============================================================
# 5. 关键发现汇总
# ============================================================
print(f"\n{'='*55}")
print("📌 关键发现")
print(f"{'='*55}")

print(f"样本量: {len(df)} 头牛")

if '最终结局' in df.columns:
    for k, v in df['最终结局'].value_counts(normalize=True).mul(100).round(1).items():
        print(f"  {k}: {v}%")

if '平均日增重' in df.columns:
    print(f"平均日增重: {df['平均日增重'].mean():.3f} kg/天")

if '总生命周期(天)' in df.columns:
    ml = df['总生命周期(天)'].mean()
    print(f"平均生命周期: {ml:.0f} 天 ({ml/30:.1f} 个月)")

if '饲料总花费_元' in df.columns:
    print(f"平均饲料花费: {df['饲料总花费_元'].mean():.0f} 元/头")

if '精料转化率' in df.columns:
    m = df['精料转化率'].median()
    print(f"精料转化率中位数: {m:.4f}")


if '累计用药次数' in df.columns and '平均日增重' in df.columns:
    sick = df[df['累计用药次数'] > 0]['平均日增重'].mean()
    healthy = df[df['累计用药次数'] == 0]['平均日增重'].mean()
    print(f"健康牛日增重: {healthy:.3f} kg/天")
    print(f"生病牛日增重: {sick:.3f} kg/天")

print(f"\n{'='*55}")
print(f"✅ 分析完成！所有图表保存在: {FIG_DIR}/")
print(f"   {'='*55}")