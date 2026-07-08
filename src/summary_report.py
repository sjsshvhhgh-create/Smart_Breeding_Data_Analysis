"""
交叉验证汇总：回归分析 × 贝叶斯估计 × 聚类分析
==============================================
目标：在相同因子条件下，三种方法结论是否一致？
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans

# ============================================================
# 0. 设置
# ============================================================
DATA_DIR = "./data/split"
FIG_DIR = "./figures/summary"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

print("=" * 70)
print("   📊 交叉验证汇总报告：三种方法 × 同一因子")
print("=" * 70)

# ============================================================
# 1. 加载数据
# ============================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# 只保留牛
for df in [train_df, val_df, test_df]:
    df.drop(df[df['牛只品种'] != 1].index, inplace=True)

# 合并全量（部分分析用全量更稳定）
all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

print(f"\n📋 数据: 训练集 {len(train_df)} | 验证集 {len(val_df)} | 测试集 {len(test_df)} | 全量 {len(all_df)}")

# ============================================================
# 2. 共同因子：初始体重
# ============================================================
print(f"\n{'─' * 70}")
print("2️⃣ 共同因子验证：初始体重 → 平均日增重")
print(f"{'─' * 70}")

COMMON_FACTOR = '初始体重'
TARGET = '平均日增重'

# 准备数据
reg_data = train_df.dropna(subset=[COMMON_FACTOR, TARGET])
X = reg_data[[COMMON_FACTOR]].values
y = reg_data[TARGET].values

# --- 方法A：回归分析 ---
print(f"\n   📐 方法A：线性回归")
lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)
# 手动算 p 值
n = len(X)
X_mat = np.c_[np.ones((n, 1)), X]
mse = np.sum((y - y_pred) ** 2) / (n - 2)
var_beta = mse * np.linalg.inv(X_mat.T @ X_mat).diagonal()
se = np.sqrt(np.abs(var_beta))[1]
t_stat = lr.coef_[0] / se
p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))
r2 = r2_score(y, y_pred)

print(f"      斜率 = {lr.coef_[0]:.6f}  (每增加1kg体重，日增重变化)")
print(f"      p值  = {p_val:.6f} {'✅ 显著' if p_val < 0.05 else '❌ 不显著'}")
print(f"      R2   = {r2:.4f}")

# --- 方法B：贝叶斯风格置信区间 ---
print(f"\n   🎲 方法B：Bootstrap 置信区间")
n_boot = 2000
boot_slopes = []
for _ in range(n_boot):
    idx = np.random.choice(n, n, replace=True)
    lr_boot = LinearRegression()
    lr_boot.fit(X[idx], y[idx])
    boot_slopes.append(lr_boot.coef_[0])
boot_slopes = np.array(boot_slopes)
ci_low, ci_high = np.percentile(boot_slopes, [2.5, 97.5])

print(f"      斜率 95% CI: [{ci_low:.6f}, {ci_high:.6f}]")
print(f"      均值 = {boot_slopes.mean():.6f}")
print(f"      {'✅ 区间不含0，显著' if ci_low > 0 or ci_high < 0 else '❌ 区间含0，不显著'}")

# --- 方法C：聚类分箱验证 ---
print(f"\n   📦 方法C：体重分箱验证")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
all_clean = all_df.dropna(subset=[COMMON_FACTOR, TARGET])
all_clean = all_clean.copy()
all_clean['体重分箱'] = kmeans.fit_predict(all_clean[[COMMON_FACTOR]].values)
bin_stats = all_clean.groupby('体重分箱').agg(
    平均体重=(COMMON_FACTOR, 'mean'),
    平均日增重=(TARGET, 'mean'),
    样本数=(COMMON_FACTOR, 'count')
).sort_values('平均体重').round(2)

print(bin_stats.to_string())
# 算趋势
if len(bin_stats) >= 2:
    trend = "上升" if bin_stats['平均日增重'].iloc[-1] > bin_stats['平均日增重'].iloc[0] else "下降"
    print(f"      趋势: 体重 {bin_stats['平均体重'].iloc[0]:.0f}→{bin_stats['平均体重'].iloc[-1]:.0f} kg, "
          f"日增重 {bin_stats['平均日增重'].iloc[0]:.3f}→{bin_stats['平均日增重'].iloc[-1]:.3f} ({trend})")

# ============================================================
# 3. 共同因子：累计用药次数
# ============================================================
print(f"\n{'─' * 70}")
print("3️⃣ 共同因子验证：累计用药次数 → 平均日增重")
print(f"{'─' * 70}")

FACTOR2 = '累计用药次数'
reg_data2 = train_df.dropna(subset=[FACTOR2, TARGET])
X2 = reg_data2[[FACTOR2]].values
y2 = reg_data2[TARGET].values
n2 = len(X2)

# 回归
lr2 = LinearRegression()
lr2.fit(X2, y2)
y_pred2 = lr2.predict(X2)
X_mat2 = np.c_[np.ones((n2, 1)), X2]
mse2 = np.sum((y2 - y_pred2) ** 2) / (n2 - 2)
var_beta2 = mse2 * np.linalg.inv(X_mat2.T @ X_mat2).diagonal()
se2 = np.sqrt(np.abs(var_beta2))[1]
t_stat2 = lr2.coef_[0] / se2
p_val2 = 2 * (1 - stats.t.cdf(np.abs(t_stat2), n2 - 2))
r2_2 = r2_score(y2, y_pred2)

print(f"\n   📐 回归: 斜率={lr2.coef_[0]:.6f}, p={p_val2:.6f}, R2={r2_2:.4f}")

# Bootstrap
boot_slopes2 = []
for _ in range(n_boot):
    idx = np.random.choice(n2, n2, replace=True)
    lr_boot = LinearRegression()
    lr_boot.fit(X2[idx], y2[idx])
    boot_slopes2.append(lr_boot.coef_[0])
boot_slopes2 = np.array(boot_slopes2)
ci_low2, ci_high2 = np.percentile(boot_slopes2, [2.5, 97.5])

print(f"   🎲 Bootstrap 95% CI: [{ci_low2:.6f}, {ci_high2:.6f}]")
print(f"   {'✅ 显著' if ci_low2 > 0 or ci_high2 < 0 else '❌ 不显著'}")

# 分箱
all_clean2 = all_df.dropna(subset=[FACTOR2, TARGET]).copy()
# 用药次数分布特殊，直接分有无用药
all_clean2['有无用药'] = (all_clean2[FACTOR2] > 0).astype(int)
med_stats = all_clean2.groupby('有无用药').agg(
    平均日增重=(TARGET, 'mean'),
    样本数=(TARGET, 'count')
).round(4)
print(f"\n   📦 分箱: 无用药 {med_stats.loc[0, '平均日增重']:.4f} (n={int(med_stats.loc[0, '样本数'])}) | "
      f"有用药 {med_stats.loc[1, '平均日增重']:.4f} (n={int(med_stats.loc[1, '样本数'])})")

# ============================================================
# 4. 共同因子：累计防疫次数
# ============================================================
print(f"\n{'─' * 70}")
print("4️⃣ 共同因子验证：累计防疫次数 → 平均日增重")
print(f"{'─' * 70}")

FACTOR3 = '累计防疫次数'
reg_data3 = train_df.dropna(subset=[FACTOR3, TARGET])
X3 = reg_data3[[FACTOR3]].values
y3 = reg_data3[TARGET].values
n3 = len(X3)

lr3 = LinearRegression()
lr3.fit(X3, y3)
y_pred3 = lr3.predict(X3)
X_mat3 = np.c_[np.ones((n3, 1)), X3]
mse3 = np.sum((y3 - y_pred3) ** 2) / (n3 - 2)
var_beta3 = mse3 * np.linalg.inv(X_mat3.T @ X_mat3).diagonal()
se3 = np.sqrt(np.abs(var_beta3))[1]
t_stat3 = lr3.coef_[0] / se3
p_val3 = 2 * (1 - stats.t.cdf(np.abs(t_stat3), n3 - 2))
r2_3 = r2_score(y3, y_pred3)

print(f"\n   📐 回归: 斜率={lr3.coef_[0]:.6f}, p={p_val3:.6f}, R2={r2_3:.4f}")

boot_slopes3 = []
for _ in range(n_boot):
    idx = np.random.choice(n3, n3, replace=True)
    lr_boot = LinearRegression()
    lr_boot.fit(X3[idx], y3[idx])
    boot_slopes3.append(lr_boot.coef_[0])
boot_slopes3 = np.array(boot_slopes3)
ci_low3, ci_high3 = np.percentile(boot_slopes3, [2.5, 97.5])

print(f"   🎲 Bootstrap 95% CI: [{ci_low3:.6f}, {ci_high3:.6f}]")
print(f"   {'✅ 显著' if ci_low3 > 0 or ci_high3 < 0 else '❌ 不显著'}")

all_clean3 = all_df.dropna(subset=[FACTOR3, TARGET]).copy()
all_clean3['有无防疫'] = (all_clean3[FACTOR3] > 0).astype(int)
vac_stats = all_clean3.groupby('有无防疫').agg(
    平均日增重=(TARGET, 'mean'),
    样本数=(TARGET, 'count')
).round(4)
print(f"\n   📦 分箱: 无防疫 {vac_stats.loc[0, '平均日增重']:.4f} (n={int(vac_stats.loc[0, '样本数'])}) | "
      f"有防疫 {vac_stats.loc[1, '平均日增重']:.4f} (n={int(vac_stats.loc[1, '样本数'])})")

# ============================================================
# 5. 聚类 × 回归交叉验证
# ============================================================
print(f"\n{'─' * 70}")
print("5️⃣ 聚类 × 回归 交叉验证：最优簇的日增重是否显著高于其他簇？")
print(f"{'─' * 70}")

# 用全量数据做聚类
cluster_data = all_df.dropna(subset=['登记月龄', '初始体重', TARGET]).copy()
cluster_data = cluster_data[(cluster_data['登记月龄'] >= 5) & (cluster_data['登记月龄'] <= 8)]
scaler = StandardScaler()
X_cluster = scaler.fit_transform(cluster_data[['登记月龄', '初始体重']])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_data['簇'] = kmeans.fit_predict(X_cluster)

# 各簇日增重
cluster_adg = cluster_data.groupby('簇')[TARGET].mean().sort_values(ascending=False)
best_cluster = cluster_adg.index[0]
worst_cluster = cluster_adg.index[-1]

print(f"\n   各簇平均日增重:")
for cid in cluster_adg.index:
    marker = ' ← 最优' if cid == best_cluster else (' ← 最差' if cid == worst_cluster else '')
    n_c = (cluster_data['簇'] == cid).sum()
    print(f"   簇 {cid}: {cluster_adg[cid]:.4f} kg/天 (n={n_c}){marker}")

# t 检验：最优簇 vs 其他
best_adg = cluster_data[cluster_data['簇'] == best_cluster][TARGET]
other_adg = cluster_data[cluster_data['簇'] != best_cluster][TARGET]
t_stat_c, p_val_c = stats.ttest_ind(best_adg, other_adg)
print(f"\n   t 检验: 最优簇 vs 其他簇")
print(f"   t = {t_stat_c:.4f}, p = {p_val_c:.6f}")
print(f"   {'✅ 显著差异' if p_val_c < 0.05 else '❌ 无显著差异'}")

# 最优簇的特征
best_center = scaler.inverse_transform(kmeans.cluster_centers_)[best_cluster]
print(f"\n   最优簇中心: 月龄={best_center[0]:.0f}月, 体重={best_center[1]:.0f}kg")

# ============================================================
# 6. 可视化：三方法一致性
# ============================================================
print(f"\n{'─' * 70}")
print("6️⃣ 生成一致性可视化")
print(f"{'─' * 70}")

# 图1：三方法对初始体重的结论对比
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 回归
ax = axes[0]
ax.scatter(X, y, alpha=0.3, s=10, color='steelblue')
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
ax.plot(x_line, lr.predict(x_line), 'r-', linewidth=2, label=f'斜率={lr.coef_[0]:.5f}')
ax.fill_between(x_line.flatten(),
                lr.predict(x_line) - 1.96 * se * (x_line.flatten() - X.mean()),
                lr.predict(x_line) + 1.96 * se * (x_line.flatten() - X.mean()),
                alpha=0.2, color='red', label='95% CI')
ax.set_xlabel('初始体重 (kg)'); ax.set_ylabel('平均日增重')
ax.set_title(f'回归分析\np={p_val:.4f} R2={r2:.3f}')
ax.legend(fontsize=8)

# Bootstrap
ax = axes[1]
ax.hist(boot_slopes, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', ls='--', linewidth=2, label='零效应线')
ax.axvline(ci_low, color='green', ls='--', label=f'95% CI')
ax.axvline(ci_high, color='green', ls='--')
ax.axvline(boot_slopes.mean(), color='orange', linewidth=2, label=f'均值={boot_slopes.mean():.5f}')
ax.set_xlabel('斜率'); ax.set_ylabel('频次')
ax.set_title(f'Bootstrap 分布\n95% CI: [{ci_low:.5f}, {ci_high:.5f}]')
ax.legend(fontsize=8)

# 分箱
ax = axes[2]
colors = plt.cm.Set2(np.linspace(0, 1, len(bin_stats)))
bars = ax.bar([f'{w:.0f}kg' for w in bin_stats['平均体重']],
              bin_stats['平均日增重'], color=colors)
ax.set_xlabel('体重分箱'); ax.set_ylabel('平均日增重')
ax.set_title('聚类分箱验证')
for bar, val in zip(bars, bin_stats['平均日增重']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'summary_01_初始体重三方法.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ summary_01_初始体重三方法.png")

# 图2：三因子一致性热力图
fig, ax = plt.subplots(figsize=(10, 4))
factors = ['初始体重', '累计用药次数', '累计防疫次数']
methods = ['回归 p值', 'Bootstrap 显著', '分箱趋势']
data = np.array([
    [p_val,       1 if ci_low > 0 or ci_high < 0 else 0,
     int(bin_stats['平均日增重'].iloc[-1] > bin_stats['平均日增重'].iloc[0])],
    [p_val2,      1 if ci_low2 > 0 or ci_high2 < 0 else 0,
     int(med_stats.loc[1, '平均日增重'] > med_stats.loc[0, '平均日增重'])],
    [p_val3,      1 if ci_low3 > 0 or ci_high3 < 0 else 0,
     int(vac_stats.loc[1, '平均日增重'] > vac_stats.loc[0, '平均日增重'])],
])

im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        if j == 0:
            text = f'{data[i, j]:.4f}'
            color = 'white' if data[i, j] > 0.5 else 'black'
        else:
            text = '✅' if data[i, j] == 1 else '❌'
            color = 'black'
        ax.text(j, i, text, ha='center', va='center', fontsize=12, color=color)

ax.set_xticks(range(3)); ax.set_xticklabels(methods)
ax.set_yticks(range(3)); ax.set_yticklabels(factors)
ax.set_title('三方法 × 三因子 一致性矩阵')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'summary_02_一致性矩阵.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ summary_02_一致性矩阵.png")

# 图3：聚类最优簇验证
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for cid in sorted(cluster_data['簇'].unique()):
    mask = cluster_data['簇'] == cid
    ax.scatter(cluster_data.loc[mask, '初始体重'], cluster_data.loc[mask, TARGET],
               alpha=0.4, s=10, label=f'簇 {cid}')
ax.set_xlabel('初始体重 (kg)'); ax.set_ylabel('平均日增重')
ax.set_title('各簇：体重 vs 日增重')
ax.legend(fontsize=8)

ax = axes[1]
cluster_adg_sorted = cluster_data.groupby('簇')[TARGET].mean().sort_values()
colors_bar = ['#2ecc71' if c == best_cluster else 'steelblue' for c in cluster_adg_sorted.index]
ax.barh([f'簇 {c}' for c in cluster_adg_sorted.index], cluster_adg_sorted.values, color=colors_bar)
ax.set_xlabel('平均日增重'); ax.set_title('聚类最优簇验证')
for i, (cid, val) in enumerate(zip(cluster_adg_sorted.index, cluster_adg_sorted.values)):
    ax.text(val + 0.001, i, f'{val:.4f}', va='center')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'summary_03_聚类验证.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ summary_03_聚类验证.png")

# ============================================================
# 7. 最终报告
# ============================================================
print(f"\n{'=' * 70}")
print("   📋 最终交叉验证报告")
print(f"{'=' * 70}")

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                     数字化赋能智慧养殖 — 交叉验证结论                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  一、初始体重（三种方法共同因子）                                      │
│     ├─ 回归分析: 斜率={lr.coef_[0]:.5f}, p={p_val:.4f}, {'显著' if p_val < 0.05 else '不显著'}          │
│     ├─ Bootstrap: 95% CI [{ci_low:.5f}, {ci_high:.5f}], {'显著' if ci_low>0 or ci_high<0 else '不显著'}    │
│     └─ 聚类分箱: 趋势={trend}                                         │
│     >> {'✅ 三种方法结论一致' if (p_val < 0.05) == (ci_low>0 or ci_high<0) else '⚠️ 方法间存在分歧'}                                   │
│                                                                     │
│  二、累计用药次数                                                     │
│     ├─ 回归分析: 斜率={lr2.coef_[0]:.5f}, p={p_val2:.4f}, {'显著' if p_val2 < 0.05 else '不显著'}          │
│     └─ Bootstrap: 95% CI [{ci_low2:.5f}, {ci_high2:.5f}], {'显著' if ci_low2>0 or ci_high2<0 else '不显著'}    │
│                                                                     │
│  三、累计防疫次数                                                     │
│     ├─ 回归分析: 斜率={lr3.coef_[0]:.5f}, p={p_val3:.4f}, {'显著' if p_val3 < 0.05 else '不显著'}          │
│     └─ Bootstrap: 95% CI [{ci_low3:.5f}, {ci_high3:.5f}], {'显著' if ci_low3>0 or ci_high3<0 else '不显著'}    │
│                                                                     │
│  四、聚类 × 回归 交叉验证                                             │
│     ├─ 最优簇日增重: {cluster_adg[best_cluster]:.4f} kg/天            │
│     ├─ 其他簇日增重: {other_adg.mean():.4f} kg/天                     │
│     ├─ t 检验: p={p_val_c:.4f}, {'显著' if p_val_c<0.05 else '不显著'}          │
│     └─ 最优特征: 月龄≈{best_center[0]:.0f}月, 体重≈{best_center[1]:.0f}kg    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

print(f"📁 图表保存在: {FIG_DIR}/")
print(f"✅ 交叉验证汇总完成！")