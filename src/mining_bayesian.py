"""
贝叶斯参数估计（PyMC 5+ 现代版）：量化不确定性与风险置信区间
================================================
v2 改进：
1. 生长样本显式过滤 (has_growth + is_adg_outlier)
2. 特征集与回归模块一致 (登记月龄、初始体重、累计用药/防疫次数)
3. 存活率对比增加小样本提示
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTENSOR_FLAGS'] = 'cxx='   # PyMC 5+ 要求

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import pymc as pm

# ============================================================
# 0. 设置与数据加载
# ============================================================
DATA_DIR = "./data/split"
FIG_DIR = "./figures/bayesian"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
try:
    az.style.use("arviz")
except Exception:
    pass

print(f"\n{'='*60}")
print("🐮 贝叶斯参数估计（PyMC）— 量化不确定性 (v2 对齐生长样本)")
print(f"{'='*60}")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# ----------------------------------------------------------
# 生长样本过滤（与 regression 模块完全一致）
# ----------------------------------------------------------
def filter_growth(df, name):
    before = len(df)
    df = df[df['has_growth'] == True].copy()
    after_has = len(df)
    if 'is_adg_outlier' in df.columns:
        df = df[df['is_adg_outlier'] == 0]
    after_outlier = len(df)
    print(f"  {name}: {before} → 有生长 {after_has} → 去异常 {after_outlier}")
    return df

print("\n📊 生长样本筛选:")
train_df = filter_growth(train_df, "训练集")
test_df  = filter_growth(test_df, "测试集")

print(f"\n训练集用于推断后验 (Train): {len(train_df)} 头牛")
print(f"测试集用于泛化验证 (Test) : {len(test_df)} 头牛\n")

# MCMC 采样参数
N_DRAWS = 2000
N_TUNE = 1000

def get_samples(trace, var_name):
    """提取后验样本数组"""
    da = trace.posterior[var_name]
    stacked = da.stack(sample=("chain", "draw"))
    return stacked.values.T if stacked.ndim > 1 else stacked.values

# ============================================================
# 1. Beta-Binomial MCMC：结局概率 (基于训练集)
# ============================================================
print(f"{'─'*60}")
print("1️⃣  结局概率的贝叶斯估计（基于 Train 学习先验经验）")
print(f"{'─'*60}")

if '最终结局' in train_df.columns:
    n_total = len(train_df)
    outcomes = ['死亡', '已出售', '存栏']
    colors = {'死亡': 'red', '已出售': 'green', '存栏': 'blue'}

    fig, ax = plt.subplots(figsize=(10, 5))

    for outcome in outcomes:
        k = (train_df['最终结局'] == outcome).sum()
        with pm.Model() as beta_model:
            p = pm.Beta('p', alpha=1, beta=1)
            obs = pm.Binomial('obs', n=n_total, p=p, observed=k)
            trace = pm.sample(N_DRAWS, tune=N_TUNE, chains=2, cores=1, random_seed=42, progressbar=False)

        p_samples = get_samples(trace, 'p')
        post_mean = p_samples.mean()
        post_hdi = az.hdi(p_samples, prob=0.95)

        print(f"\n   🎯 {outcome}: {k}/{n_total} ({k/n_total*100:.1f}%)")
        print(f"      后验均值: {post_mean:.4f} ({post_mean*100:.2f}%)")
        print(f"      95% HDI:   [{post_hdi[0]:.4f}, {post_hdi[1]:.4f}]")

        sns.kdeplot(x=p_samples, color=colors[outcome], linewidth=2, fill=True, alpha=0.2,
                    ax=ax, label=f'{outcome} (均值={post_mean:.3f})')

    ax.set_title('结局概率的后验分布 (训练集推断)')
    ax.set_xlabel('概率 p')
    ax.set_ylabel('后验密度')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'bayes_01_结局概率后验.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n   📈 bayes_01_结局概率后验.png ✅")

# ============================================================
# 2. Normal MCMC：平均日增重 (基于训练集)
# ============================================================
print(f"\n{'─'*60}")
print("2️⃣  平均日增重的贝叶斯估计（基于 Train）")
print(f"{'─'*60}")

if '平均日增重' in train_df.columns:
    adg_train = train_df['平均日增重'].dropna().values

    with pm.Model() as normal_model:
        mu = pm.Normal('mu', mu=0.8, sigma=0.3)
        sigma = pm.HalfNormal('sigma', sigma=0.3)
        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=adg_train)
        trace = pm.sample(N_DRAWS, tune=N_TUNE, chains=2, cores=1, random_seed=42, progressbar=False)

    mu_samples = get_samples(trace, 'mu')
    mu_mean = mu_samples.mean()
    mu_hdi = az.hdi(mu_samples, prob=0.95)

    print(f"\n   🎯 训练集牛群平均日增重推断")
    print(f"      后验均值: {mu_mean:.3f} kg/天")
    print(f"      95% HDI:   [{mu_hdi[0]:.3f}, {mu_hdi[1]:.3f}] kg/天")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.kdeplot(x=mu_samples, color='crimson', linewidth=2, fill=True, alpha=0.3, ax=axes[0])
    axes[0].axvline(adg_train.mean(), color='gray', linestyle='--', label=f'样本均值={adg_train.mean():.3f}')
    axes[0].set_title('平均日增重的后验分布')
    axes[0].set_xlabel('日增重 (kg/天)')
    axes[0].legend()

    with normal_model:
        ppc = pm.sample_posterior_predictive(trace, random_seed=42, progressbar=False)
    ppc_samples = ppc.posterior_predictive['obs'].values.flatten()
    axes[1].hist(adg_train, bins=30, density=True, alpha=0.5, color='steelblue', label='实际观测(Train)')
    sns.kdeplot(x=ppc_samples, color='crimson', linewidth=2, fill=True, alpha=0.2, ax=axes[1], label='后验预测')
    axes[1].set_title('后验预测检验（模型拟合检查）')
    axes[1].set_xlabel('日增重 (kg/天)')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'bayes_02_日增重贝叶斯.png'), dpi=150, bbox_inches='tight')
    plt.close()

    az.plot_trace(trace)
    fig = plt.gcf()
    fig.suptitle('MCMC 采样轨迹（诊断收敛性）', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'bayes_02_trace_诊断.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   📈 bayes_02_trace_诊断.png ✅")

# ============================================================
# 3. 贝叶斯线性回归：影响因子分析 (Train 学习, Test 验证)
# ============================================================
print(f"\n{'─'*60}")
print("3️⃣  贝叶斯线性回归：影响日增重的因子分析 (Train推断 -> Test泛化)")
print(f"{'─'*60}")

if '平均日增重' in train_df.columns:
    # 特征与回归模块完全一致
    features = ['登记月龄', '初始体重', '累计用药次数', '累计防疫次数']
    existing_features = [f for f in features if f in train_df.columns]

    if len(existing_features) >= 2:
        # 训练集
        train_valid = train_df[existing_features + ['平均日增重']].dropna()
        X_train = train_valid[existing_features].values
        y_train = train_valid['平均日增重'].values

        # 标准化
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_std[X_std == 0] = 1
        X_scaled = (X_train - X_mean) / X_std
        y_mean = y_train.mean()
        y_std = y_train.std()
        y_scaled = (y_train - y_mean) / y_std

        # MCMC
        with pm.Model() as lm_model:
            alpha = pm.Normal('alpha', mu=0, sigma=1)
            betas = pm.Normal('beta', mu=0, sigma=1, shape=len(existing_features))
            sigma = pm.HalfNormal('sigma', sigma=1)
            mu_pred = alpha + pm.math.dot(X_scaled, betas)
            obs = pm.Normal('obs', mu=mu_pred, sigma=sigma, observed=y_scaled)
            trace_lm = pm.sample(N_DRAWS, tune=N_TUNE, chains=2, cores=1, random_seed=42, progressbar=False)

        alpha_samples = get_samples(trace_lm, 'alpha')
        beta_samples = get_samples(trace_lm, 'beta')

        # 还原到原始量纲
        betas_original = beta_samples * y_std / X_std
        alpha_original = alpha_samples * y_std + y_mean - np.dot(betas_original, X_mean)

        print(f"\n      {'因子':15s} {'后验均值':10s} {'标准差':10s} {'95% HDI':20s}")
        print(f"      {'─'*60}")

        fig, axes = plt.subplots(1, len(existing_features), figsize=(5*len(existing_features), 4))
        if len(existing_features) == 1:
            axes = [axes]

        for i, (feat, ax) in enumerate(zip(existing_features, axes)):
            b_samples = betas_original[:, i]
            b_mean = b_samples.mean()
            b_hdi = az.hdi(b_samples, prob=0.95)
            print(f"      {feat:15s} {b_mean:+.4f}     {b_samples.std():.4f}   [{b_hdi[0]:+.4f}, {b_hdi[1]:+.4f}]")

            sns.kdeplot(x=b_samples, color='steelblue', linewidth=2, fill=True, alpha=0.3, ax=ax)
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
            ax.axvline(b_hdi[0], color='gray', linestyle=':', alpha=0.5)
            ax.axvline(b_hdi[1], color='gray', linestyle=':', alpha=0.5)
            ax.set_title(f'{feat}\n均值={b_mean:.3f}')

        plt.suptitle('贝叶斯回归系数后验分布 (基于训练集)', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'bayes_03_回归系数后验.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n   📈 bayes_03_回归系数后验.png ✅")

        # 预测评估
        beta_mean_final = betas_original.mean(axis=0)
        alpha_mean_final = alpha_original.mean()

        y_pred_train = alpha_mean_final + np.dot(X_train, beta_mean_final)
        ss_res_train = ((y_train - y_pred_train) ** 2).sum()
        ss_tot_train = ((y_train - y_train.mean()) ** 2).sum()
        r2_train = 1 - ss_res_train / ss_tot_train
        print(f"      🎯 训练集 R² (In-sample): {r2_train:.4f}")

        # 测试集
        test_valid = test_df[existing_features + ['平均日增重']].dropna()
        X_test = test_valid[existing_features].values
        y_test = test_valid['平均日增重'].values
        if len(y_test) > 0:
            y_pred_test = alpha_mean_final + np.dot(X_test, beta_mean_final)
            ss_res_test = ((y_test - y_pred_test) ** 2).sum()
            ss_tot_test = ((y_test - y_test.mean()) ** 2).sum()
            r2_test = 1 - ss_res_test / ss_tot_test
            print(f"      🛡️ 测试集 R² (Out-of-sample): {r2_test:.4f}")

# ============================================================
# 4. 两组对比：重牛 vs 轻牛存活率 (基于训练集)
# ============================================================
print(f"\n{'─'*60}")
print("4️⃣  两组对比：重牛 vs 轻牛存活率差异 (基于 Train)")
print(f"{'─'*60}")

if '初始体重' in train_df.columns and '最终结局' in train_df.columns:
    heavy = (train_df['初始体重'] >= 350).values
    survived = (train_df['最终结局'] != '死亡').values

    n_heavy, n_light = heavy.sum(), (~heavy).sum()
    survived_heavy, survived_light = (heavy & survived).sum(), (~heavy & survived).sum()
    n_dead_train = (train_df['最终结局'] == '死亡').sum()

    print(f"   训练集中死亡牛数: {n_dead_train}，结论受小样本影响较大，需谨慎解读。")

    with pm.Model() as compare_model:
        p_heavy = pm.Beta('p_heavy', alpha=2, beta=2)
        p_light = pm.Beta('p_light', alpha=2, beta=2)
        obs_h = pm.Binomial('obs_h', n=n_heavy, p=p_heavy, observed=survived_heavy)
        obs_l = pm.Binomial('obs_l', n=n_light, p=p_light, observed=survived_light)
        diff = pm.Deterministic('diff', p_heavy - p_light)
        trace_cmp = pm.sample(N_DRAWS, tune=N_TUNE, chains=2, cores=1, random_seed=42, progressbar=False)

    diff_samples = get_samples(trace_cmp, 'diff')
    p_heavy_samples = get_samples(trace_cmp, 'p_heavy')
    p_light_samples = get_samples(trace_cmp, 'p_light')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax_idx, (p_samples, color, label) in enumerate([(p_heavy_samples, 'green', '重牛'), (p_light_samples, 'orange', '轻牛')]):
        sns.kdeplot(x=p_samples, color=color, linewidth=2, fill=True, alpha=0.2, ax=axes[ax_idx])
        axes[ax_idx].set_title(f'{label}\n均值={p_samples.mean():.3f}')

    sns.kdeplot(x=diff_samples, color='purple', linewidth=2, fill=True, alpha=0.3, ax=axes[2])
    axes[2].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[2].set_title(f'存活率差异\nP(差异>0)={(diff_samples > 0).mean():.2%}')

    diff_mean = diff_samples.mean()
    diff_hdi = az.hdi(diff_samples, prob=0.95)
    prob_heavy_better = (diff_samples > 0).mean()
    print(f"\n   🎯 存活率差异 (重牛 - 轻牛)")
    print(f"      后验均值: {diff_mean:.4f} ({diff_mean*100:.2f}%)")
    print(f"      95% HDI:   [{diff_hdi[0]:.4f}, {diff_hdi[1]:.4f}]")
    print(f"      P(重牛存活率 > 轻牛存活率) = {prob_heavy_better:.4f}")

    plt.suptitle('重牛 vs 轻牛存活率对比 (训练集)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'bayes_04_两组存活率对比.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   📈 bayes_04_两组存活率对比.png ✅")

print(f"\n{'='*60}")
print("✅ 贝叶斯分析完成！图表保存在 ./figures/bayesian/")
print(f"{'='*60}")