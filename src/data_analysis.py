import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# 牛只全生命周期数据分析脚本（分层聚焦版）
# ============================================================

DATA_DIR = "./data/processed"
FIG_DIR = "./figures/data_analysis"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

# ------------------------------------------------------------
# 0. 加载宽表 + 异常过滤 + 数据分层
# ------------------------------------------------------------
df = pd.read_csv(os.path.join(DATA_DIR, "牛只全生命周期分析宽表.csv"))
print(f"\n{'='*55}")
print(f"🐮 牛只全生命周期分析报告（分层聚焦）")
print(f"{'='*55}")
print(f"原始数据: {len(df)} 头牛\n")

# 定义分层子集
# 全量
df_all = df.copy()
# 生长分析子集：has_growth=True 且 非异常
df_growth = df[(df['has_growth'] == True) & (df['is_adg_outlier'] == 0)].copy()
# 成本分析子集：has_cost=True
df_cost = df[df['has_cost'] == True].copy()

print(f"📊 数据分层:")
print(f"   全量存栏: {len(df_all)} 头")
print(f"   生长分析可用: {len(df_growth)} 头 (有有效生长记录)")
print(f"   成本分析可用: {len(df_cost)} 头 (同时有饲喂+生长记录)")
print(f"   已出售牛只: {df['最终结局'].value_counts().get('已出售', 0)} 头")

# ============================================================
# 1. 全量存栏结构（无缺失的核心分类）
# ============================================================
print(f"\n{'='*55}")
print("📌 一、全量存栏结构")
print(f"{'='*55}")

# 结局分布
print("\n【最终结局分布】")
for k, v in df_all['最终结局'].value_counts().items():
    print(f"   {k}: {v} 头 ({v/len(df_all)*100:.1f}%)")

# 防疫覆盖率（全量）
if '累计防疫次数' in df_all.columns:
    vacc_cover = (df_all['累计防疫次数'] > 0).sum()
    print(f"\n【防疫覆盖率】 {vacc_cover}/{len(df_all)} = {vacc_cover/len(df_all)*100:.1f}%")

# 引入渠道分布（有渠道信息的牛）
if '引入渠道名称' in df_all.columns:
    channel_cnt = df_all['引入渠道名称'].notna().sum()
    print(f"\n【引入渠道已知比例】 {channel_cnt}/{len(df_all)} = {channel_cnt/len(df_all)*100:.1f}%")
    if channel_cnt > 0:
        print(df_all['引入渠道名称'].value_counts().to_string())

# 已出售牛的出栏概况
df_sold = df_all[df_all['最终结局'] == '已出售']
if not df_sold.empty:
    print(f"\n【已出售牛只概况】 共 {len(df_sold)} 头")
    if '总生命周期(天)' in df_sold.columns:
        print(f"   平均养殖周期: {df_sold['总生命周期(天)'].mean():.0f} 天 ({df_sold['总生命周期(天)'].mean()/30:.1f} 个月)")
    if '出售重量' in df_sold.columns:
        print(f"   平均出售体重: {df_sold['出售重量'].mean():.1f} kg")
    if '出售单价' in df_sold.columns:
        print(f"   平均出售单价: {df_sold['出售单价'].mean():.1f} 元/kg")
    if '总金额' in df_sold.columns:
        print(f"   平均出售总价: {df_sold['总金额'].mean():.0f} 元")

# ============================================================
# 2. 生长样本分析（513头）
# ============================================================
if len(df_growth) > 0:
    print(f"\n{'='*55}")
    print(f"📈 二、生长分析 (n={len(df_growth)})")
    print(f"{'='*55}")

    print("\n【日增重统计】")
    print(df_growth['平均日增重'].describe().round(3).to_string())

    # 2.1 日增重分布图
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df_growth['平均日增重'], bins=30, kde=True, color='steelblue')
    ax.set_title(f'平均日增重分布 (n={len(df_growth)})')
    ax.set_xlabel('kg/天')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'growth_dist.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   图表: growth_dist.png ✅")

    # 2.2 初始体重 vs 日增重
    if '初始体重' in df_growth.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=df_growth, x='初始体重', y='平均日增重', alpha=0.5, ax=ax)
        sns.regplot(data=df_growth, x='初始体重', y='平均日增重', scatter=False, ax=ax, color='red')
        ax.set_title('初始体重 vs 日增重')
        ax.set_xlabel('初始体重 (kg)')
        ax.set_ylabel('平均日增重 (kg/天)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'growth_weight_scatter.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("   图表: growth_weight_scatter.png ✅")

    # 2.3 引入渠道对比（如果数据足够）
    if '引入渠道名称' in df_growth.columns:
        channel_growth = df_growth.dropna(subset=['引入渠道名称'])
        # 只显示样本量>5的渠道
        channel_counts = channel_growth['引入渠道名称'].value_counts()
        valid_channels = channel_counts[channel_counts >= 5].index
        channel_growth = channel_growth[channel_growth['引入渠道名称'].isin(valid_channels)]
        if len(channel_growth) > 0 and channel_growth['引入渠道名称'].nunique() > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=channel_growth, x='引入渠道名称', y='平均日增重', palette='Set2')
            ax.set_title('各引入渠道日增重对比 (样本≥5)')
            ax.set_xlabel('引入渠道')
            ax.set_ylabel('日增重 (kg/天)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, 'growth_channel_box.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("   图表: growth_channel_box.png ✅")

    # 2.4 防疫次数与日增重
    if '累计防疫次数' in df_growth.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df_growth, x='累计防疫次数', y='平均日增重', palette='viridis')
        ax.set_title('防疫次数 vs 日增重')
        ax.set_xlabel('累计防疫次数')
        ax.set_ylabel('日增重 (kg/天)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'growth_vaccine_box.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("   图表: growth_vaccine_box.png ✅")

# ============================================================
# 3. 成本效益样本分析（106头）
# ============================================================
if len(df_cost) > 0:
    print(f"\n{'='*55}")
    print(f"💰 三、成本效益分析 (n={len(df_cost)})")
    print(f"{'='*55}")

    # 基本统计
    if '每公斤增重成本' in df_cost.columns:
        print(f"每公斤增重成本: 均值 {df_cost['每公斤增重成本'].mean():.2f} 元/kg, 中位数 {df_cost['每公斤增重成本'].median():.2f} 元/kg")

    if '总饲料转化率' in df_cost.columns:
        print(f"总饲料转化率 (增重/饲料): 均值 {df_cost['总饲料转化率'].mean():.3f}, 中位数 {df_cost['总饲料转化率'].median():.3f}")

    # 计算已出售牛的盈亏
    df_cost_sold = df_cost[df_cost['最终结局'] == '已出售']
    if not df_cost_sold.empty:
        print(f"\n【已出售牛成本效益 (n={len(df_cost_sold)})】")
        if '出栏盈亏' in df_cost_sold.columns:
            print(f"   平均出栏盈亏: {df_cost_sold['出栏盈亏'].mean():.0f} 元")
            profit_rate = (df_cost_sold['出栏盈亏'] > 0).mean() * 100
            print(f"   盈利比例: {profit_rate:.0f}%")
        if '每公斤增重成本' in df_cost_sold.columns:
            print(f"   平均每公斤增重成本: {df_cost_sold['每公斤增重成本'].mean():.2f} 元/kg")

    # 3.1 成本分布图
    if '每公斤增重成本' in df_cost.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_cost['每公斤增重成本'].dropna(), bins=20, kde=True, color='coral')
        ax.set_title(f'每公斤增重成本分布 (n={len(df_cost)})')
        ax.set_xlabel('元/kg')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'cost_per_kg_dist.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("   图表: cost_per_kg_dist.png ✅")

    # 3.2 日增重 vs 每公斤增重成本
    if '平均日增重' in df_cost.columns and '每公斤增重成本' in df_cost.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=df_cost, x='平均日增重', y='每公斤增重成本', alpha=0.7)
        ax.set_title('日增重 vs 每公斤增重成本')
        ax.set_xlabel('日增重 (kg/天)')
        ax.set_ylabel('每公斤增重成本 (元)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'cost_vs_adg.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("   图表: cost_vs_adg.png ✅")

    # 3.3 出栏盈亏分布（已出售牛）
    if '出栏盈亏' in df_cost_sold.columns and not df_cost_sold.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_cost_sold['出栏盈亏'], bins=15, kde=True, color='green')
        ax.axvline(0, color='red', linestyle='--')
        ax.set_title(f'出栏盈亏分布 (n={len(df_cost_sold)})')
        ax.set_xlabel('盈亏 (元)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'cost_profit_dist.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("   图表: cost_profit_dist.png ✅")

# ============================================================
# 4. 总结关键发现
# ============================================================
print(f"\n{'='*55}")
print("📌 关键发现摘要")
print(f"{'='*55}")
print(f"1. 数据断层: 仅19%的牛有生长记录，4%有饲喂成本记录。分析需分层进行。")
if len(df_growth) > 0:
    print(f"2. 生长分析 (n={len(df_growth)}): 平均日增重 {df_growth['平均日增重'].mean():.3f} kg/天。")
    if '初始体重' in df_growth.columns:
        corr = df_growth['初始体重'].corr(df_growth['平均日增重'])
        print(f"   初始体重与日增重相关系数: {corr:.3f}")
if len(df_cost) > 0:
    print(f"3. 成本分析 (n={len(df_cost)}): 每公斤增重成本中位数 {df_cost['每公斤增重成本'].median():.2f} 元。")
    if not df_cost_sold.empty:
        profit_ratio = (df_cost_sold['出栏盈亏'] > 0).mean() * 100
        print(f"   已出售牛中盈利比例: {profit_ratio:.0f}%")
print(f"\n✅ 分析完成！图表保存在: {FIG_DIR}/")
print(f"{'='*55}")