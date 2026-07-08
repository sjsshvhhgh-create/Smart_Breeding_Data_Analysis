"""
数据挖掘模块三：聚类分析 — 科学分群与"牛犊选择"
==================================================
严格遵循《目标》流程：
  - 训练集：拟合 K-Means，学习簇中心
  - 验证集：调参（选最优 k），不更新模型权重
  - 测试集：全程不参与训练/调参，仅做最终泛化评估

正向分类：用入栏时已知的 登记月龄 + 初始体重 分群
后验复盘：对比各群体在全生命周期中的 总饲料转化率、最终结局、平均日增重
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ============================================================
# 0. 基础设置
# ============================================================
DATA_DIR = "./data/split"
FIG_DIR = "./figures/clustering"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

print(f"\n{'='*65}")
print("🔬 模块三：聚类分析 — 正向分群 + 后验复盘")
print(f"{'='*65}")

# ============================================================
# 1. 加载三份数据集
# ============================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# 只保留牛（品种=1），排除长白猪、蒙古羊等非牛数据
train_df = train_df[train_df['牛只品种'] == 1]
val_df   = val_df[val_df['牛只品种'] == 1]
test_df  = test_df[test_df['牛只品种'] == 1]

print(f"✅ 数据加载: 训练集 {len(train_df)} | 验证集 {len(val_df)} | 测试集 {len(test_df)}")

# ============================================================
# 2. 特征工程：入栏时就能知道的特征
# ============================================================
CLUSTER_FEATURES = ['登记月龄', '初始体重']
# 后验评估特征（聚类完成后才用，不参与训练）
EVAL_FEATURES = ['总饲料转化率', '平均日增重', '总生命周期(天)', '最终结局']

print(f"\n📋 聚类特征（入栏时可知）: {CLUSTER_FEATURES}")
print(f"📋 后验评估特征         : {EVAL_FEATURES}")

# ============================================================
# 3. 预处理：标准化
# ============================================================
print(f"\n{'─'*65}")
print("3️⃣ 数据预处理")
print(f"{'─'*65}")

def prepare_data(df):
    """剔除缺失值，返回干净的 DataFrame"""
    return df.dropna(subset=CLUSTER_FEATURES).copy()

train_clean = prepare_data(train_df)
val_clean   = prepare_data(val_df)
test_clean  = prepare_data(test_df)

print(f"   有效样本: 训练集 {len(train_clean)} | 验证集 {len(val_clean)} | 测试集 {len(test_clean)}")

# 在训练集上拟合 Scaler，然后变换三集
scaler = StandardScaler()
X_train = scaler.fit_transform(train_clean[CLUSTER_FEATURES])
X_val   = scaler.transform(val_clean[CLUSTER_FEATURES])
X_test  = scaler.transform(test_clean[CLUSTER_FEATURES])

print(f"   训练集月龄范围: {train_clean['登记月龄'].min():.0f} ~ {train_clean['登记月龄'].max():.0f} 月")
print(f"   训练集体重范围: {train_clean['初始体重'].min():.0f} ~ {train_clean['初始体重'].max():.0f} kg")

# ============================================================
# 4. 训练集拟合 + 验证集调参：选最优 k
# ============================================================
print(f"\n{'─'*65}")
print("4️⃣ 训练集拟合 K-Means + 验证集调参（选最优 k）")
print(f"{'─'*65}")

K_RANGE = range(2, 9)

results = []
for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_train)

    train_labels = kmeans.predict(X_train)
    train_silhouette = silhouette_score(X_train, train_labels)

    val_labels = kmeans.predict(X_val)
    val_silhouette = silhouette_score(X_val, val_labels)
    val_davies = davies_bouldin_score(X_val, val_labels)
    val_ch = calinski_harabasz_score(X_val, val_labels)

    results.append({
        'k': k,
        'train_silhouette': train_silhouette,
        'val_silhouette': val_silhouette,
        'val_davies_bouldin': val_davies,
        'val_calinski_harabasz': val_ch,
        'model': kmeans,
    })

results_df = pd.DataFrame(results)

print(f"\n   {'k':>3s} {'训练轮廓':>10s} {'验证轮廓':>10s} {'DB指数':>10s} {'CH指数':>10s}")
print(f"   {'─'*55}")
for _, r in results_df.iterrows():
    print(f"   {r['k']:>3d} {r['train_silhouette']:>10.4f} {r['val_silhouette']:>10.4f} "
          f"{r['val_davies_bouldin']:>10.4f} {r['val_calinski_harabasz']:>10.1f}")

# 综合评分选最优 k
results_df['silhouette_rank'] = results_df['val_silhouette'].rank(ascending=False)
results_df['davies_rank'] = results_df['val_davies_bouldin'].rank(ascending=True)
results_df['ch_rank'] = results_df['val_calinski_harabasz'].rank(ascending=False)
results_df['综合排名'] = (results_df['silhouette_rank'] +
                         results_df['davies_rank'] +
                         results_df['ch_rank'])

best_k_row = results_df.loc[results_df['综合排名'].idxmin()]
best_k = int(best_k_row['k'])
best_model = results_df.loc[results_df['k'] == best_k, 'model'].values[0]

print(f"\n   🏆 最优 k = {best_k} (验证集综合排名最优)")
print(f"      轮廓系数 = {best_k_row['val_silhouette']:.4f}")
print(f"      DB 指数  = {best_k_row['val_davies_bouldin']:.4f}")
print(f"      CH 指数  = {best_k_row['val_calinski_harabasz']:.1f}")

# ============================================================
# 5. 测试集最终盲测
# ============================================================
print(f"\n{'─'*65}")
print(f"5️⃣ 测试集最终盲测（k={best_k}）")
print(f"{'─'*65}")

test_labels = best_model.predict(X_test)
test_silhouette = silhouette_score(X_test, test_labels)
test_davies = davies_bouldin_score(X_test, test_labels)
test_ch = calinski_harabasz_score(X_test, test_labels)

print(f"   测试集轮廓系数: {test_silhouette:.4f}")
print(f"   测试集 DB 指数:  {test_davies:.4f}")
print(f"   测试集 CH 指数:  {test_ch:.1f}")

# ============================================================
# 6. 后验复盘：全量数据定论
# ============================================================
print(f"\n{'─'*65}")
print("6️⃣ 后验复盘：各群体全生命周期表现对比")
print(f"{'─'*65}")

all_clean = pd.concat([train_clean, val_clean, test_clean], ignore_index=True)
X_all = scaler.transform(all_clean[CLUSTER_FEATURES])
all_clean = all_clean.copy()
all_clean['簇'] = best_model.predict(X_all)

# 6.1 各簇入栏特征画像
print(f"\n📊 各簇入栏特征画像（均值）:")
cluster_profile = all_clean.groupby('簇')[CLUSTER_FEATURES].agg(['mean', 'std', 'count']).round(2)
print(cluster_profile.to_string())

# 6.2 各簇总饲料转化率
if '总饲料转化率' in all_clean.columns:
    print(f"\n📊 各簇总饲料转化率（越低越好）:")
    fcr_stats = all_clean.groupby('簇')['总饲料转化率'].agg(['mean', 'median', 'std', 'count']).round(4)
    fcr_stats = fcr_stats.dropna()
    print(fcr_stats.to_string())
    best_fcr = fcr_stats['mean'].idxmin()
    worst_fcr = fcr_stats['mean'].idxmax()
    print(f"   🏆 转化率最优: 簇 {best_fcr} (均值={fcr_stats.loc[best_fcr, 'mean']:.4f})")
    print(f"   ⚠️  转化率最差: 簇 {worst_fcr} (均值={fcr_stats.loc[worst_fcr, 'mean']:.4f})")

# 6.3 各簇平均日增重
if '平均日增重' in all_clean.columns:
    print(f"\n📊 各簇平均日增重（越高越好）:")
    adg_stats = all_clean.groupby('簇')['平均日增重'].agg(['mean', 'median', 'std', 'count']).round(4)
    adg_stats = adg_stats.dropna()
    print(adg_stats.to_string())
    best_adg = adg_stats['mean'].idxmax()
    print(f"   🏆 日增重最优: 簇 {best_adg} (均值={adg_stats.loc[best_adg, 'mean']:.4f} kg/天)")

# 6.4 各簇最终结局分布
if '最终结局' in all_clean.columns:
    print(f"\n📊 各簇最终结局分布:")
    outcome_crosstab = pd.crosstab(
        all_clean['簇'], all_clean['最终结局'], normalize='index'
    ).round(4) * 100
    print(outcome_crosstab.to_string())
    if '死亡' in outcome_crosstab.columns:
        death_risk = outcome_crosstab['死亡']
        safest = death_risk.idxmin()
        riskiest = death_risk.idxmax()
        print(f"   🛡️ 最安全: 簇 {safest} (死亡率={death_risk[safest]:.1f}%)")
        print(f"   🔴 最高风险: 簇 {riskiest} (死亡率={death_risk[riskiest]:.1f}%)")

# 6.5 各簇总生命周期
if '总生命周期(天)' in all_clean.columns:
    print(f"\n📊 各簇养殖周期（天）:")
    life_stats = all_clean.groupby('簇')['总生命周期(天)'].agg(['mean', 'median', 'count']).round(1)
    life_stats = life_stats.dropna()
    print(life_stats.to_string())

# ============================================================
# 7. 综合评分：选出最优牛犊
# ============================================================
print(f"\n{'─'*65}")
print("7️⃣ 牛犊综合评分：买牛指南")
print(f"{'─'*65}")

all_clean['评分'] = 0
# 转化率（越低越好，权重 40%）
if '总饲料转化率' in all_clean.columns:
    fcr = all_clean.groupby('簇')['总饲料转化率'].transform('mean')
    fcr_range = fcr.max() - fcr.min()
    if fcr_range > 0:
        all_clean['评分'] += (1 - (fcr - fcr.min()) / fcr_range) * 40
# 死亡率（越低越好，权重 40%）
if '最终结局' in all_clean.columns:
    death_rate = all_clean.groupby('簇')['最终结局'].transform(
        lambda x: (x == '死亡').mean()
    )
    all_clean['评分'] += (1 - death_rate) * 40
# 日增重（越高越好，权重 20%）
if '平均日增重' in all_clean.columns:
    adg = all_clean.groupby('簇')['平均日增重'].transform('mean')
    adg_range = adg.max() - adg.min()
    if adg_range > 0:
        all_clean['评分'] += ((adg - adg.min()) / adg_range) * 20

cluster_scores = all_clean.groupby('簇')['评分'].mean().sort_values(ascending=False)

print(f"\n   综合评分排名（转化率40% + 存活率40% + 日增重20%）:")
for i, (cid, score) in enumerate(cluster_scores.items()):
    medal = ['🥇', '🥈', '🥉'][i] if i < 3 else f'  {i+1}.'
    prof = cluster_profile.loc[cid]
    print(f"   {medal} 簇 {cid}: 综合分={score:.1f}")
    print(f"      月龄={prof[('登记月龄','mean')]:.0f}±{prof[('登记月龄','std')]:.0f}月 | "
          f"体重={prof[('初始体重','mean')]:.0f}±{prof[('初始体重','std')]:.0f}kg | "
          f"样本量={int(prof[('登记月龄','count')])}头")

# ============================================================
# 8. 可视化
# ============================================================
print(f"\n{'─'*65}")
print("8️⃣ 生成可视化图表")
print(f"{'─'*65}")

# 图1：肘部法则 + 轮廓系数
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

inertias = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_train)
    inertias.append(km.inertia_)

ax = axes[0]
ax.plot(K_RANGE, inertias, 'o-', color='steelblue', linewidth=2)
ax.axvline(best_k, color='red', ls='--', alpha=0.7, label=f'最优 k={best_k}')
ax.set_xlabel('簇数 k'); ax.set_ylabel('惯性 (Inertia)')
ax.set_title('肘部法则：寻找最优 k'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(results_df['k'], results_df['train_silhouette'], 'o-', color='steelblue', label='训练集')
ax.plot(results_df['k'], results_df['val_silhouette'], 's--', color='orange', label='验证集')
ax.axvline(best_k, color='red', ls='--', alpha=0.7, label=f'最优 k={best_k}')
ax.set_xlabel('簇数 k'); ax.set_ylabel('轮廓系数')
ax.set_title('轮廓系数：越高越好'); ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'cluster_01_肘部法则与轮廓系数.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ cluster_01_肘部法则与轮廓系数.png")

# 图2：原始空间散点图（月龄 vs 体重，按簇着色）
fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.tab10(np.linspace(0, 1, best_k))
for c in range(best_k):
    mask = all_clean['簇'] == c
    ax.scatter(all_clean.loc[mask, '登记月龄'], all_clean.loc[mask, '初始体重'],
               c=[colors[c]], alpha=0.5, s=20, label=f'簇 {c} ({mask.sum()}头)')
# 标注簇中心
centers = scaler.inverse_transform(best_model.cluster_centers_)
ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=300,
           edgecolor='black', linewidth=2, label='簇中心', zorder=5)
ax.set_xlabel('登记月龄（月）'); ax.set_ylabel('初始体重（kg）')
ax.set_title(f'牛群聚类结果：月龄 vs 体重 (k={best_k})')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'cluster_02_月龄体重散点图.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ cluster_02_月龄体重散点图.png")

# 图3：各簇转化率箱线图
if '总饲料转化率' in all_clean.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data = all_clean.dropna(subset=['总饲料转化率'])
    sns.boxplot(data=plot_data, x='簇', y='总饲料转化率', palette='Set2', ax=ax)
    sns.stripplot(data=plot_data, x='簇', y='总饲料转化率', color='black', alpha=0.2, size=3, ax=ax)
    ax.set_title('各簇总饲料转化率分布（越低越好）')
    ax.set_xlabel('簇'); ax.set_ylabel('总饲料转化率')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'cluster_03_转化率对比.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ cluster_03_转化率对比.png")

# 图4：各簇最终结局堆叠柱状图
if '最终结局' in all_clean.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    crosstab = pd.crosstab(all_clean['簇'], all_clean['最终结局'])
    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    crosstab_pct.plot(kind='bar', stacked=True, ax=ax,
                      color=['#2ecc71', '#e74c3c', '#3498db'])
    ax.set_title('各簇最终结局分布')
    ax.set_xlabel('簇'); ax.set_ylabel('占比 (%)')
    ax.legend(title='结局')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'cluster_04_结局分布.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ cluster_04_结局分布.png")

# 图5：各簇日增重箱线图
if '平均日增重' in all_clean.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data = all_clean.dropna(subset=['平均日增重'])
    sns.boxplot(data=plot_data, x='簇', y='平均日增重', palette='Set2', ax=ax)
    sns.stripplot(data=plot_data, x='簇', y='平均日增重', color='black', alpha=0.2, size=3, ax=ax)
    ax.set_title('各簇平均日增重分布（越高越好）')
    ax.set_xlabel('簇'); ax.set_ylabel('平均日增重 (kg/天)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'cluster_05_日增重对比.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ cluster_05_日增重对比.png")

# 图6：综合评分排名
fig, ax = plt.subplots(figsize=(9, 5))
colors_bar = ['gold' if i == 0 else 'silver' if i == 1 else '#cd7f32' if i == 2 else 'steelblue'
              for i in range(len(cluster_scores))]
bars = ax.bar([f'簇 {int(c)}' for c in cluster_scores.index],
              cluster_scores.values, color=colors_bar)
ax.set_title('牛犊综合评分排名（越高越值得买）')
ax.set_ylabel('综合评分')
for bar, score in zip(bars, cluster_scores.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{score:.1f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'cluster_06_综合评分排名.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ cluster_06_综合评分排名.png")

# ============================================================
# 9. 总结
# ============================================================
print(f"\n{'='*65}")
print("✅ 聚类分析完成！")
print(f"{'='*65}")
print(f"\n   📁 图表保存在: {FIG_DIR}/")
print(f"   🔢 最优簇数: k = {best_k}")
print(f"   📊 测试集轮廓系数: {test_silhouette:.4f}")

best_cluster_id = cluster_scores.index[0]
best_prof = cluster_profile.loc[best_cluster_id]
print(f"\n   🥇 推荐购买特征:")
print(f"      登记月龄 ≈ {best_prof[('登记月龄','mean')]:.0f} ± {best_prof[('登记月龄','std')]:.0f} 月")
print(f"      初始体重 ≈ {best_prof[('初始体重','mean')]:.0f} ± {best_prof[('初始体重','std')]:.0f} kg")
print(f"\n{'='*65}")