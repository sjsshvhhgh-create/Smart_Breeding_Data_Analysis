"""
数据挖掘模块三：聚类分析 — 科学分群与"牛犊选择"（修正版）
==========================================================
修正要点：
1. 限定 5-8 月龄犊牛，聚焦牛犊选择
2. 显式过滤 has_growth + is_adg_outlier，确保生长数据有效
3. 训练/验证集复盘形成购买建议，测试集严格盲测（仅做最终差异检验）
4. 评分体系分层：日增重+出栏率（主） / 饲料转化率（辅，仅106头）
5. 各簇渠道分布事后透视
6. 死亡样本极少，评分降权并加注说明
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
from scipy import stats

# ============================================================
# 0. 基础设置
# ============================================================
DATA_DIR = "./data/split"
FIG_DIR = "./figures/clustering"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

print(f"\n{'='*65}")
print("🔬 模块三：聚类分析 — 犊牛科学分群 (5-8月龄)")
print(f"{'='*65}")

# ============================================================
# 1. 加载并准备数据
# ============================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

CLUSTER_FEATURES = ['登记月龄', '初始体重']          # 入栏时可知
EVAL_FEATURES = ['平均日增重', '总饲料转化率', '总生命周期(天)', '最终结局']  # 后验评估

def prepare_calf_data(df, name):
    """筛选可用于犊牛聚类的样本：5-8月龄 + 有效生长记录 + 无异常"""
    before = len(df)
    # 限定牛只（品种=1）
    if '牛只品种' in df.columns:
        df = df[df['牛只品种'] == 1].copy()
    # 有效生长数据
    df = df[(df['has_growth'] == True) & (df['is_adg_outlier'] == 0)]
    after_growth = len(df)
    # 限定犊牛月龄
    df = df[(df['登记月龄'] >= 5) & (df['登记月龄'] <= 8)]
    after_month = len(df)
    # 删除聚类特征缺失
    df = df.dropna(subset=CLUSTER_FEATURES)
    after_dropna = len(df)
    print(f"  {name}: {before} → 生长有效 {after_growth} → 5-8月龄 {after_month} → 无缺失 {after_dropna}")
    return df

print("\n📊 数据准备:")
train_clean = prepare_calf_data(train_df, "训练集")
val_clean   = prepare_calf_data(val_df,   "验证集")
test_clean  = prepare_calf_data(test_df,  "测试集")

print(f"\n   最终聚类样本: 训练 {len(train_clean)} | 验证 {len(val_clean)} | 测试 {len(test_clean)}")

# ============================================================
# 2. 标准化与模型训练/调参
# ============================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(train_clean[CLUSTER_FEATURES])
X_val   = scaler.transform(val_clean[CLUSTER_FEATURES])
X_test  = scaler.transform(test_clean[CLUSTER_FEATURES])

K_RANGE = range(2, 8)
results = []

for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_train)
    val_labels = kmeans.predict(X_val)
    # 验证集指标
    sil = silhouette_score(X_val, val_labels)
    db = davies_bouldin_score(X_val, val_labels)
    ch = calinski_harabasz_score(X_val, val_labels)
    results.append({'k': k, 'val_silhouette': sil, 'val_davies': db, 'val_ch': ch, 'model': kmeans})

results_df = pd.DataFrame(results)
# 综合排名选最优k
results_df['rank_sil'] = results_df['val_silhouette'].rank(ascending=False)
results_df['rank_db']  = results_df['val_davies'].rank(ascending=True)
results_df['rank_ch']  = results_df['val_ch'].rank(ascending=False)
results_df['total_rank'] = results_df[['rank_sil','rank_db','rank_ch']].sum(axis=1)
best_idx = results_df['total_rank'].idxmin()
best_k = int(results_df.loc[best_idx, 'k'])
best_model = results_df.loc[best_idx, 'model']

print(f"\n   最优簇数 k = {best_k} (综合轮廓系数、DB、CH)")
print(f"   验证集轮廓: {results_df.loc[best_idx,'val_silhouette']:.3f} | DB: {results_df.loc[best_idx,'val_davies']:.3f} | CH: {results_df.loc[best_idx,'val_ch']:.1f}")

# ============================================================
# 3. 分配标签（训练+验证用于复盘，测试仅保留标签）
# ============================================================
train_clean['cluster'] = best_model.predict(X_train)
val_clean['cluster']   = best_model.predict(X_val)
test_clean['cluster']  = best_model.predict(X_test)

# 训练+验证合集（后验复盘）
train_val = pd.concat([train_clean, val_clean], ignore_index=True)

# ============================================================
# 4. 后验复盘（仅在训练+验证集上进行）
# ============================================================
print(f"\n{'─'*65}")
print("4️⃣ 后验复盘：各簇生长表现 (训练+验证集)")
print(f"{'─'*65}")

# 4.1 基本画像
profile = train_val.groupby('cluster').agg(
    样本量=('平均日增重', 'count'),
    平均月龄=('登记月龄', 'mean'),
    平均初始体重=('初始体重', 'mean'),
    平均日增重=('平均日增重', 'mean'),
    日增重标准差=('平均日增重', 'std')
).round(3)
print("\n簇基本画像:")
print(profile.to_string())

# 4.2 日增重差异显著性（最优 vs 最差）
best_c = profile['平均日增重'].idxmax()
worst_c = profile['平均日增重'].idxmin()
best_vals = train_val[train_val['cluster'] == best_c]['平均日增重']
worst_vals = train_val[train_val['cluster'] == worst_c]['平均日增重']
if len(best_vals) > 2 and len(worst_vals) > 2:
    t, p = stats.ttest_ind(best_vals, worst_vals)
    print(f"\n最优簇 {best_c} vs 最差簇 {worst_c} 日增重 t检验: t={t:.2f}, p={p:.4f}")

# 4.3 饲料转化率（仅限有饲喂记录的牛）
if '总饲料转化率' in train_val.columns:
    fcr_data = train_val.dropna(subset=['总饲料转化率'])
    if len(fcr_data) > 10:
        fcr_stats = fcr_data.groupby('cluster')['总饲料转化率'].agg(['mean', 'count']).round(4)
        print(f"\n各簇饲料转化率 (仅 {len(fcr_data)} 头有记录):")
        print(fcr_stats.to_string())
    else:
        print("\n⚠️ 有饲料转化率的样本过少，无法按簇统计")

# 4.4 最终结局（出栏率，因死亡极少故改称出栏率）
outcome_tbl = pd.crosstab(train_val['cluster'], train_val['最终结局'], normalize='index') * 100
print("\n各簇结局分布 (%):")
print(outcome_tbl.round(1).to_string())
# 出栏率（已出售比例）
if '已出售' in outcome_tbl.columns:
    print("\n各簇出栏率:")
    for c in outcome_tbl.index:
        print(f"  簇 {c}: {outcome_tbl.loc[c, '已出售']:.1f}%")

# 4.5 渠道分布
if '引入渠道名称' in train_val.columns:
    channel_dist = pd.crosstab(train_val['cluster'], train_val['引入渠道名称'], normalize='index') * 100
    print("\n各簇渠道分布 (%):")
    print(channel_dist.round(1).to_string())

# ============================================================
# 5. 综合评分（仅基于训练+验证集）
# ============================================================
print(f"\n{'─'*65}")
print("5️⃣ 犊牛购买推荐评分")
print(f"{'─'*65}")

score_df = profile[['平均日增重', '样本量']].copy()
# 出栏率
if '已出售' in outcome_tbl.columns:
    score_df['出栏率'] = outcome_tbl['已出售'] / 100.0
else:
    score_df['出栏率'] = 1.0

# 归一化并加权
def norm(x): return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0

score_df['score_adg'] = norm(score_df['平均日增重']) * 60
score_df['score_out'] = norm(score_df['出栏率']) * 40
score_df['total_score'] = score_df['score_adg'] + score_df['score_out']
score_df = score_df.sort_values('total_score', ascending=False)

print("综合评分 (日增重60% + 出栏率40%):")
for c, row in score_df.iterrows():
    print(f"  簇 {c}: 总分 {row['total_score']:.1f}  (日增重 {row['平均日增重']:.3f} kg/d, 出栏率 {row['出栏率']*100:.1f}%)")

best_final = score_df.index[0]
best_info = profile.loc[best_final]
print(f"\n🥇 推荐购买犊牛特征 (簇 {best_final}):")
print(f"   月龄 ≈ {best_info['平均月龄']:.1f} 月, 体重 ≈ {best_info['平均初始体重']:.0f} kg")
print(f"   预期日增重: {best_info['平均日增重']:.3f} kg/天")

# ============================================================
# 6. 测试集最终验证 (仅做最优vs最差对比)
# ============================================================
print(f"\n{'─'*65}")
print("6️⃣ 测试集盲测验证")
print(f"{'─'*65}")

test_best = test_clean[test_clean['cluster'] == best_final]['平均日增重']
test_worst = test_clean[test_clean['cluster'] == score_df.index[-1]]['平均日增重']
if len(test_best) > 2 and len(test_worst) > 2:
    t_test, p_test = stats.ttest_ind(test_best, test_worst)
    print(f"   推荐簇 {best_final} vs 最差簇: t={t_test:.2f}, p={p_test:.4f}")
else:
    print("   测试集样本不足，无法检验")

test_sil = silhouette_score(X_test, test_clean['cluster'])
print(f"   测试集轮廓系数: {test_sil:.3f}")

# ============================================================
# 7. 可视化 (选取关键图表)
# ============================================================
# 肘部法则 + 轮廓系数
fig, ax = plt.subplots(1, 2, figsize=(14,5))
inertias = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_train)
    inertias.append(km.inertia_)
ax[0].plot(K_RANGE, inertias, 'o-')
ax[0].axvline(best_k, color='red', ls='--', label=f'最优 k={best_k}')
ax[0].set_title('肘部法则'); ax[0].set_xlabel('k'); ax[0].set_ylabel('惯性')
ax[0].legend()
ax[1].plot(results_df['k'], results_df['val_silhouette'], 'o-', color='orange')
ax[1].axvline(best_k, color='red', ls='--')
ax[1].set_title('验证集轮廓系数'); ax[1].set_xlabel('k'); ax[1].set_ylabel('轮廓系数')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'cluster_01_k_select.png'), dpi=150, bbox_inches='tight')
plt.close()

# 散点图 (训练+验证)
fig, ax = plt.subplots(figsize=(10,7))
colors = plt.cm.tab10(np.linspace(0,1,best_k))
for c in range(best_k):
    subset = train_val[train_val['cluster'] == c]
    ax.scatter(subset['登记月龄'], subset['初始体重'], c=[colors[c]], alpha=0.6, label=f'簇{c} ({len(subset)}头)')
centers = scaler.inverse_transform(best_model.cluster_centers_)
ax.scatter(centers[:,0], centers[:,1], c='red', marker='X', s=300, edgecolors='k', label='簇中心')
ax.set_xlabel('登记月龄'); ax.set_ylabel('初始体重 (kg)')
ax.set_title(f'犊牛聚类结果 (k={best_k})')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'cluster_02_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()

# 日增重箱线图 (训练+验证)
fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(data=train_val, x='cluster', y='平均日增重', palette='Set2', ax=ax)
ax.set_title('各簇日增重分布 (训练+验证集)')
ax.set_xlabel('簇'); ax.set_ylabel('日增重 (kg/天)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'cluster_03_adg_box.png'), dpi=150, bbox_inches='tight')
plt.close()

# 结局堆叠柱状图
if '最终结局' in train_val.columns:
    fig, ax = plt.subplots(figsize=(10,5))
    ctab = pd.crosstab(train_val['cluster'], train_val['最终结局'], normalize='index')*100
    ctab.plot(kind='bar', stacked=True, ax=ax, color=['#2ecc71','#e74c3c','#3498db'])
    ax.set_title('各簇结局分布'); ax.set_xlabel('簇'); ax.set_ylabel('百分比')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'cluster_04_outcome.png'), dpi=150, bbox_inches='tight')
    plt.close()

# 综合评分图
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar([f'簇 {c}' for c in score_df.index], score_df['total_score'], color='steelblue')
ax.set_title('犊牛购买推荐评分 (日增重60% + 出栏率40%)')
ax.set_ylabel('评分')
for bar, s in zip(bars, score_df['total_score']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1, f'{s:.1f}', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'cluster_05_score.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ 所有图表已保存至 {FIG_DIR}/")
print(f"{'='*65}\n")