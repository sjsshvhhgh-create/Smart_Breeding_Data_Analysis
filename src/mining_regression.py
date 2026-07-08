"""
数据挖掘模块一：多元回归分析（日增重预测与饲喂效益最优化）
======================================================
v3 改进：
1. 特征改为高覆盖率指标（初始体重/用药/防疫/生命周期），样本量从65→500+
2. 纯 sklearn/numpy，无 statsmodels 依赖
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
FIG_DIR = "./figures/regression"  # 回归模块单独目录
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

print(f"\n{'='*65}")
print("📈 模块一：多元回归分析（促长因子 + 边际效益）")
print(f"{'='*65}")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))

# 高覆盖率特征（每头牛都有，样本量 ~500+）
FEATURES = [
    '初始体重',
    '累计用药次数',
    '累计防疫次数',
    '总生命周期(天)',
]
TARGET = '平均日增重'

train_clean = train_df.dropna(subset=FEATURES + [TARGET])
test_clean = test_df.dropna(subset=FEATURES + [TARGET])
val_clean = val_df.dropna(subset=FEATURES + [TARGET])

print(f"✅ 有效样本: 训练集 {len(train_clean)} | 测试集 {len(test_clean)} | 验证集 {len(val_clean)}\n")

# ============================================================
# 1. 模型拟合与三集 R2
# ============================================================
print(f"{'─'*65}")
print("1️⃣ 模型泛化能力 (R2)")
print(f"{'─'*65}")

X_train, y_train = train_clean[FEATURES], train_clean[TARGET]
X_test, y_test = test_clean[FEATURES], test_clean[TARGET]
X_val, y_val = val_clean[FEATURES], val_clean[TARGET]

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

train_r2 = r2_score(y_train, lr_model.predict(X_train))
val_r2 = r2_score(y_val, lr_model.predict(X_val))
test_r2 = r2_score(y_test, lr_model.predict(X_test))

print(f"  训练集 R2 = {train_r2:.4f}  (学习)")
print(f"  验证集 R2 = {val_r2:.4f}   (中期观察)")
print(f"  测试集 R2 = {test_r2:.4f}   (最终盲测)")

if val_r2 < 0 or test_r2 < 0:
    print(f"  ⚠️  R2 为负 → 模型预测能力弱于直接用均值，可能特征不足或数据噪声大")
print()

# ============================================================
# 2. 显著性检验（手写 p 值）
# ============================================================
print(f"{'─'*65}")
print("2️⃣ 促长因子敏感度与显著性")
print(f"{'─'*65}")

n = len(X_train)
p = len(FEATURES)
X_mat = np.c_[np.ones((n, 1)), X_train.values]
y_pred = lr_model.predict(X_train)
mse = np.sum((y_train - y_pred) ** 2) / (n - p - 1)

try:
    var_beta = mse * np.linalg.inv(np.dot(X_mat.T, X_mat)).diagonal()
    se_beta = np.sqrt(np.abs(var_beta))[1:]  # 剔除截距
    t_stats = lr_model.coef_ / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
except np.linalg.LinAlgError:
    p_values = np.full(p, np.nan)

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
    if r['p值'] < 0.05:
        note = f"{direction}（显著）"
    else:
        note = f"{direction}（不显著）"
    print(f"  {r['特征']:15s} {r['系数']:+10.4f} {r['p值']:8.4f}  {r['显著性']:6s}  {note}")

# ============================================================
# 3. 标准化系数排名
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

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：标准化系数
ax = axes[0]
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in std_df['标准化系数']]
ax.barh(std_df['特征'], std_df['标准化系数'], color=colors)
ax.axvline(0, color='black', lw=1)
ax.set_title('特征重要性排名（标准化系数）')
ax.set_xlabel('标准化影响程度')

# 右：实际 vs 预测
ax = axes[1]
y_test_pred = lr_model.predict(X_test)
ax.scatter(y_test, y_test_pred, alpha=0.5, c='steelblue')
lims = [min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())]
ax.plot(lims, lims, 'r--', alpha=0.6)
ax.set_xlabel('实际日增重'); ax.set_ylabel('预测日增重')
ax.set_title(f'测试集：实际 vs 预测 (R2={test_r2:.3f})')
ax.set_xlim(lims); ax.set_ylim(lims)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'reg_01_特征重要性.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  ✅ reg_01_特征重要性.png")

# ============================================================
# 4. 边际效益与最优点
# ============================================================
print(f"\n{'─'*65}")
print("4️⃣ 边际效益：饲喂投入 vs 日增重")
print(f"{'─'*65}")

econ = train_df.dropna(subset=['日均花费_元', '平均日增重'])
econ = econ[(econ['日均花费_元'] > 0) & (econ['平均日增重'] > 0)]

if len(econ) >= 20:
    x_cost = econ['日均花费_元'].values
    y_gain = econ['平均日增重'].values
    a, b, c = np.polyfit(x_cost, y_gain, 2)
    print(f"  拟合曲线: y = {a:.4f}x² + {b:.2f}x + {c:.2f}  (n={len(econ)})")

    if a < 0:
        opt_cost = -b / (2 * a)
        opt_gain = a * opt_cost**2 + b * opt_cost + c
        print(f"  最优日均花费: {opt_cost:.2f} 元/天 → 日增重 {opt_gain:.3f} kg/天")
        print(f"  净利润: {opt_gain * 30 - opt_cost:.2f} 元/天 (活牛30元/kg)")

        plt.figure(figsize=(10, 5))
        plt.scatter(x_cost, y_gain, alpha=0.3, s=15, c='gray', label='真实数据')
        x_line = np.linspace(x_cost.min(), x_cost.max(), 100)
        y_line = a * x_line**2 + b * x_line + c
        plt.plot(x_line, y_line, 'r-', lw=2.5, label='拟合曲线')
        plt.axvline(opt_cost, color='green', ls='--', label=f'最优点 {opt_cost:.2f}元/天')
        plt.scatter([opt_cost], [opt_gain], c='green', s=150, marker='*', zorder=5)
        plt.fill_between(x_line, y_line.min(), y_line, where=x_line <= opt_cost,
                         color='green', alpha=0.1, label='递增区')
        plt.fill_between(x_line, y_line.min(), y_line, where=x_line > opt_cost,
                         color='red', alpha=0.1, label='递减区')
        plt.title('饲喂边际效益与经济最优点')
        plt.xlabel('日均饲喂花费 (元/天)')
        plt.ylabel('平均日增重 (kg/天)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'reg_02_边际效益最优点.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ reg_02_边际效益最优点.png")
    else:
        print(f"  ⚠️ 二次项为正，未出现边际递减")
else:
    print(f"  ⚠️ 有效数据不足 (n={len(econ)})，跳过")

print(f"\n{'='*65}")
print(f"✅ 分析完成！图表保存在 {FIG_DIR}/")
print(f"{'='*65}")