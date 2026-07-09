"""
数据挖掘模块一：多元回归分析（日增重预测 + 饲喂效益探索）
========================================================
改进：
1. 特征限定为入栏时可观察指标（登记月龄、初始体重）
   加上养殖过程中可积累的指标（用药、防疫次数）
2. 明确分层：生长样本（has_growth=True & 非异常）→ 预测日增重
              成本样本（has_cost=True）→ 探索饲喂投入与日增重关系
3. 三集评估 + p值显著性 + 标准化系数排名
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# 0. 基础设置
# ============================================================
DATA_DIR = "./data/split"
FIG_DIR = "./figures/regression"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

print(f"\n{'='*65}")
print("📈 模块一：多元回归分析（促长因子 + 饲喂效益探索）")
print(f"{'='*65}")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))

# ============================================================
# 1. 生长样本过滤：只用有有效生长记录的牛
# ============================================================
def filter_growth_sample(df, name=""):
    """筛选可用于日增重建模的样本"""
    before = len(df)
    df = df[df['has_growth'] == True].copy()
    after_has = len(df)
    if 'is_adg_outlier' in df.columns:
        df = df[df['is_adg_outlier'] == 0]
    after_outlier = len(df)
    print(f"  {name}: {before} → 有生长记录 {after_has} → 去异常 {after_outlier}")
    return df

print("\n📊 生长样本筛选:")
train_g = filter_growth_sample(train_df, "训练集")
val_g = filter_growth_sample(val_df, "验证集")
test_g = filter_growth_sample(test_df, "测试集")

# ============================================================
# 2. 特征与目标
# ============================================================
FEATURES = [
    '登记月龄',
    '初始体重',
    '累计用药次数',
    '累计防疫次数',
]
TARGET = '平均日增重'

# 去除含有缺失值的行（在特征或目标上）
train_clean = train_g.dropna(subset=FEATURES + [TARGET])
val_clean = val_g.dropna(subset=FEATURES + [TARGET])
test_clean = test_g.dropna(subset=FEATURES + [TARGET])

print(f"\n📊 最终建模样本: 训练 {len(train_clean)} | 验证 {len(val_clean)} | 测试 {len(test_clean)}")

X_train, y_train = train_clean[FEATURES], train_clean[TARGET]
X_val, y_val = val_clean[FEATURES], val_clean[TARGET]
X_test, y_test = test_clean[FEATURES], test_clean[TARGET]

# ============================================================
# 3. 模型拟合与三集 R2
# ============================================================
print(f"\n{'─'*65}")
print("1️⃣ 模型泛化能力 (R2)")
print(f"{'─'*65}")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

train_r2 = r2_score(y_train, lr_model.predict(X_train))
val_r2 = r2_score(y_val, lr_model.predict(X_val))
test_r2 = r2_score(y_test, lr_model.predict(X_test))

print(f"  训练集 R2 = {train_r2:.4f}")
print(f"  验证集 R2 = {val_r2:.4f}")
print(f"  测试集 R2 = {test_r2:.4f}")

if val_r2 < 0 or test_r2 < 0:
    print("  ⚠️ R2 为负，模型预测能力弱于直接用均值，特征解释力有限")

# ============================================================
# 4. 显著性检验（手工计算 p 值）
# ============================================================
print(f"\n{'─'*65}")
print("2️⃣ 促长因子敏感度与显著性")
print(f"{'─'*65}")

n = len(X_train)
p = len(FEATURES)
X_mat = np.c_[np.ones((n, 1)), X_train.values]
y_pred = lr_model.predict(X_train)
mse = np.sum((y_train - y_pred) ** 2) / (n - p - 1)

try:
    var_beta = mse * np.linalg.inv(np.dot(X_mat.T, X_mat)).diagonal()
    se_beta = np.sqrt(np.abs(var_beta))[1:]
    t_stats = lr_model.coef_ / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
except np.linalg.LinAlgError:
    p_values = np.full(p, np.nan)
    print("  ⚠️ 矩阵奇异，无法计算 p 值")

coef_df = pd.DataFrame({
    '特征': FEATURES,
    '系数': lr_model.coef_,
    'p值': p_values,
})
coef_df['显著性'] = coef_df['p值'].apply(
    lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else 'ns'))
)
coef_df = coef_df.sort_values('系数', ascending=False)

print(f"  {'特征':15s} {'系数':>10s} {'p值':>8s}  {'显著性':>6s}  {'解读'}")
print(f"  {'─'*55}")
for _, r in coef_df.iterrows():
    direction = '📈 促进' if r['系数'] > 0 else '📉 抑制'
    note = f"{direction}（显著）" if r['p值'] < 0.05 else f"{direction}（不显著）"
    print(f"  {r['特征']:15s} {r['系数']:+10.4f} {r['p值']:8.4f}  {r['显著性']:6s}  {note}")

# ============================================================
# 5. 标准化系数排名
# ============================================================
print(f"\n{'─'*65}")
print("3️⃣ 特征重要性排名（标准化系数）")
print(f"{'─'*65}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
lr_std = LinearRegression()
lr_std.fit(X_train_s, y_train)

std_df = pd.DataFrame({
    '特征': FEATURES,
    '标准化系数': lr_std.coef_,
}).sort_values('标准化系数', key=abs, ascending=False)

for i, (_, row) in enumerate(std_df.iterrows()):
    arrow = '📈' if row['标准化系数'] > 0 else '📉'
    print(f"  {i+1}. {row['特征']:15s} {arrow} {row['标准化系数']:+.4f}")

# 可视化：标准化系数 + 实际 vs 预测
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：标准化系数
ax = axes[0]
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in std_df['标准化系数']]
ax.barh(std_df['特征'], std_df['标准化系数'], color=colors)
ax.axvline(0, color='black', lw=1)
ax.set_title('特征重要性排名（标准化系数）')
ax.set_xlabel('标准化影响程度')

# 右：实际 vs 预测（测试集）
ax = axes[1]
y_test_pred = lr_model.predict(X_test)
ax.scatter(y_test, y_test_pred, alpha=0.5, c='steelblue')
lims = [min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())]
ax.plot(lims, lims, 'r--', alpha=0.6)
ax.set_xlabel('实际日增重 (kg/天)')
ax.set_ylabel('预测日增重 (kg/天)')
ax.set_title(f'测试集：实际 vs 预测 (R2={test_r2:.3f})')
ax.set_xlim(lims)
ax.set_ylim(lims)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'reg_01_特征重要性.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  ✅ reg_01_特征重要性.png")

# ============================================================
# 6. 边际效益探索（保守分析，仅线性 + 散点）
# ============================================================
print(f"\n{'─'*65}")
print("4️⃣ 饲喂投入与日增重关系探索（成本样本：n≈106）")
print(f"{'─'*65}")

# 从全量中提取成本样本（不分集，探索性质）
df_all = pd.read_csv("./data/processed/牛只全生命周期分析宽表.csv")
cost_sample = df_all[df_all['has_cost'] == True].copy()
cost_sample = cost_sample.dropna(subset=['日均花费_元', '平均日增重'])
cost_sample = cost_sample[cost_sample['日均花费_元'] > 0]

if len(cost_sample) >= 20:
    x_cost = cost_sample['日均花费_元'].values
    y_gain = cost_sample['平均日增重'].values
    corr = np.corrcoef(x_cost, y_gain)[0, 1]
    print(f"  有效样本: {len(cost_sample)} 头")
    print(f"  日均花费与日增重相关系数: {corr:.3f}")

    # 线性回归参考
    lin_reg = LinearRegression()
    lin_reg.fit(x_cost.reshape(-1, 1), y_gain)
    lin_r2 = r2_score(y_gain, lin_reg.predict(x_cost.reshape(-1, 1)))
    print(f"  线性拟合 R2: {lin_r2:.4f}  (斜率: {lin_reg.coef_[0]:.4f})")

    # 散点图 + 线性趋势线
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_cost, y_gain, alpha=0.5, c='steelblue', edgecolors='grey', linewidth=0.5)
    x_line = np.linspace(x_cost.min(), x_cost.max(), 100)
    y_line = lin_reg.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line, 'r-', lw=2, label=f'线性拟合 (R2={lin_r2:.3f})')
    ax.set_xlabel('日均饲喂花费 (元/天)')
    ax.set_ylabel('平均日增重 (kg/天)')
    ax.set_title(f'饲喂投入 vs 日增重 (n={len(cost_sample)})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'reg_02_饲喂投入散点.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ reg_02_饲喂投入散点.png")

    # 若线性关系不显著，不输出最优点
    if corr < 0.1:
        print("  ℹ️ 日均花费与日增重几乎无线性关系，无法给出最优花费建议。")
    else:
        print("  ℹ️ 仅展示趋势，样本量有限，不作经济最优点外推。")
else:
    print(f"  ⚠️ 成本分析有效样本不足 (n={len(cost_sample)})，跳过")

print(f"\n{'='*65}")
print(f"✅ 多元回归分析完成！图表保存在 {FIG_DIR}/")
print(f"{'='*65}")