import pandas as pd
import numpy as np
import os

# 设定已清洗 CSV 文件的路径
DATA_DIR = "./data/processed"

def load_csv(filename):
    """辅助函数：安全读取 CSV，防止某些表完全为空时报错"""
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"警告：未找到文件 {filename}")
        return pd.DataFrame()

print("正在读取核心 CSV 表格...")
df_base = load_csv("资产基础表.csv")
df_growth = load_csv("成长信息表.csv")
df_med = load_csv("用药信息表.csv")
df_vaccine = load_csv("防疫信息表.csv")
df_sale = load_csv("出售信息表.csv")
df_dead = load_csv("死亡信息表.csv")
df_feed_detail = load_csv("喂养记录明细表.csv")
df_feed_group_asset = load_csv("喂养分组资产表.csv")

# 打印各表列名，便于调试
print("\n资产基础表列名:", df_base.columns.tolist())
print("成长信息表列名:", df_growth.columns.tolist() if not df_growth.empty else "空表")
print("用药信息表列名:", df_med.columns.tolist() if not df_med.empty else "空表")
print("防疫信息表列名:", df_vaccine.columns.tolist() if not df_vaccine.empty else "空表")
print("喂养记录明细表列名:", df_feed_detail.columns.tolist() if not df_feed_detail.empty else "空表")
print("喂养分组资产表列名:", df_feed_group_asset.columns.tolist() if not df_feed_group_asset.empty else "空表")

# ----------------------------------------------------
# 1. 整理主表：牛只基础属性
# ----------------------------------------------------
column_mapping = {
    '自增ID': '资产 ID',
    '耳标编号': '智能耳标编号',
    '品种，字典值西门塔尔、安格斯等注：品种和资产种类关联（有级联关系）': '牛只品种',
    '性别，公/母': '性别',
    '登记月龄，以月为单位': '登记月龄',
    '出生年月，yyyyMMdd通过月龄计算出生年月（当前月-初始月龄）': '出生日期',
    '登记体重': '初始体重',
    '资产状态，0存栏1出售2死亡': '当前状态',
    '引入渠道ID': '引入渠道ID',
    '引入渠道名称': '引入渠道名称',
}

available_cols = [col for col in column_mapping.keys() if col in df_base.columns]
missing_cols = [col for col in column_mapping.keys() if col not in df_base.columns]
if missing_cols:
    print(f"\n[警告] 资产基础表中以下列不存在，将被跳过: {missing_cols}")

df_main = df_base[available_cols].rename(columns=column_mapping).drop_duplicates(subset=['资产 ID']).copy()

# ---- 样本筛选瀑布图追踪 ----
waterfall = [('原始资产基础表（去重后）', len(df_main))]

# ---- 数据总览诊断 ----
main_ids_set = set(df_main['资产 ID'].unique())
print(f"\n{'='*50}")
print(f"📊 资产基础表: {len(df_main)} 头牛")
print(f"   资产ID范围: {min(main_ids_set)} ~ {max(main_ids_set)}")
if '租户ID' in df_base.columns:
    print(f"   租户分布: {df_base['租户ID'].value_counts().to_dict()}")

# ----------------------------------------------------
# 2. 聚合特征：生长指标（核心目标变量）
# ----------------------------------------------------
if not df_growth.empty:
    df_growth = df_growth.rename(columns={'资产ID': '资产 ID', '资产体重': '体重', '记录日期': '测量日期'})

    growth_unique_ids = set(df_growth['资产 ID'].unique())
    growth_matched = growth_unique_ids & main_ids_set
    print(f"\n📊 成长信息表: {len(df_growth)} 条记录, {len(growth_unique_ids)} 头牛")
    print(f"   资产ID范围: {min(growth_unique_ids)} ~ {max(growth_unique_ids)}")
    print(f"   匹配主表: {len(growth_matched)} 头, 未匹配: {len(growth_unique_ids) - len(growth_matched)} 头")

    # 计算日增重
    df_growth = df_growth.sort_values(['资产 ID', '测量日期']).reset_index(drop=True)
    df_growth['上次体重'] = df_growth.groupby('资产 ID')['体重'].shift(1)
    df_growth['上次日期'] = df_growth.groupby('资产 ID')['测量日期'].shift(1)

    # 天数差
    df_growth['天数差'] = (
        pd.to_datetime(df_growth['测量日期']) - pd.to_datetime(df_growth['上次日期'])
    ).dt.days

    # 初始化日增重为 NaN，避免 0 干扰均值
    df_growth['日增重'] = np.nan
    mask = (df_growth['上次体重'].notna() &
            df_growth['上次日期'].notna() &
            df_growth['天数差'] > 0)  # 排除同日记录
    df_growth.loc[mask, '日增重'] = (
        (df_growth.loc[mask, '体重'] - df_growth.loc[mask, '上次体重']) /
        df_growth.loc[mask, '天数差']
    )

    # 聚合生长特征：最新体重取最后一次测量值，防止异常高值
    df_growth_last = df_growth.sort_values('测量日期').groupby('资产 ID').last()
    growth_features = df_growth.groupby('资产 ID').agg(
        平均日增重=('日增重', 'mean'),
        成长记录次数=('资产 ID', 'count')
    ).reset_index()
    growth_features = growth_features.merge(
        df_growth_last[['体重']].rename(columns={'体重': '最新体重'}).reset_index(),
        on='资产 ID', how='left'
    )

    df_main = pd.merge(df_main, growth_features, on='资产 ID', how='left')
    waterfall.append(('有生长记录', df_main['平均日增重'].notna().sum()))

# ----------------------------------------------------
# 3. 聚合特征：用药与疾病风险
# ----------------------------------------------------
if not df_med.empty:
    df_med = df_med.rename(columns={'资产ID': '资产 ID'})
    med_features = df_med.groupby('资产 ID').agg(
        累计用药次数=('资产 ID', 'count')
    ).reset_index()
    df_main = pd.merge(df_main, med_features, on='资产 ID', how='left')
    df_main['累计用药次数'] = df_main['累计用药次数'].fillna(0)

# ----------------------------------------------------
# 4. 聚合特征：防疫接种情况
# ----------------------------------------------------
if not df_vaccine.empty:
    df_vaccine = df_vaccine.rename(columns={'资产ID': '资产 ID'})
    vaccine_features = df_vaccine.groupby('资产 ID').agg(
        累计防疫次数=('资产 ID', 'count')
    ).reset_index()
    df_main = pd.merge(df_main, vaccine_features, on='资产 ID', how='left')
    df_main['累计防疫次数'] = df_main['累计防疫次数'].fillna(0)

# ----------------------------------------------------
# 5. 衍生特征：总增重
# ----------------------------------------------------
if '最新体重' in df_main.columns and '初始体重' in df_main.columns:
    df_main['总增重'] = df_main['最新体重'] - df_main['初始体重']

# ----------------------------------------------------
# 6. 聚合特征：饲料消耗（按头汇总）
# ----------------------------------------------------
if not df_feed_detail.empty:
    print("正在整合饲料消耗信息...")
    feed_mapping = {
        '资产ID': '资产 ID',
        '喂养日期': '喂养日期',
        '精料总喂养量，单位：千克': '精料量_kg',
        '草料总喂养量，单位：千克': '草料量_kg',
        '酒糟总喂养量，单位：千克': '酒糟量_kg',
        '总花费，单位：元': '饲料花费_元',
    }
    available_feed_cols = [col for col in feed_mapping.keys() if col in df_feed_detail.columns]
    df_feed = df_feed_detail[available_feed_cols].rename(columns=feed_mapping)

    # 汇总：总饲料量、总花费、喂养天数（去重自然天数）
    feed_features = df_feed.groupby('资产 ID').agg(
        精料总量_kg=('精料量_kg', 'sum'),
        草料总量_kg=('草料量_kg', 'sum'),
        酒糟总量_kg=('酒糟量_kg', 'sum'),
        饲料总花费_元=('饲料花费_元', 'sum'),
        喂养天数=('喂养日期', 'nunique'),   # 去重后的自然天数
    ).reset_index()

    # 计算日均指标（总量 ÷ 天数）
    feed_features['日均精料_kg'] = feed_features['精料总量_kg'] / feed_features['喂养天数']
    feed_features['日均草料_kg'] = feed_features['草料总量_kg'] / feed_features['喂养天数']
    feed_features['日均酒糟_kg'] = feed_features['酒糟总量_kg'] / feed_features['喂养天数']
    feed_features['日均花费_元'] = feed_features['饲料总花费_元'] / feed_features['喂养天数']

    df_main = pd.merge(df_main, feed_features, on='资产 ID', how='left')

    # 饲料转化率
    if '总增重' in df_main.columns and '精料总量_kg' in df_main.columns:
        mask_fcr = df_main['精料总量_kg'].notna() & (df_main['精料总量_kg'] > 0) & df_main['总增重'].notna()
        df_main.loc[mask_fcr, '精料转化率'] = df_main.loc[mask_fcr, '总增重'] / df_main.loc[mask_fcr, '精料总量_kg']

        df_main['总饲料_kg'] = df_main['精料总量_kg'].fillna(0) + df_main['草料总量_kg'].fillna(0) + df_main['酒糟总量_kg'].fillna(0)
        mask_fcr_total = (df_main['总饲料_kg'] > 0) & df_main['总增重'].notna()
        df_main.loc[mask_fcr_total, '总饲料转化率'] = df_main.loc[mask_fcr_total, '总增重'] / df_main.loc[mask_fcr_total, '总饲料_kg']

    waterfall.append(('有喂养记录', df_main['喂养天数'].notna().sum()))

# ----------------------------------------------------
# 7. 聚合特征：出售信息
# ----------------------------------------------------
if not df_sale.empty:
    print("正在整合出售信息...")
    sale_mapping = {
        '资产ID': '资产 ID',
        '出售日期': '出售日期',
        '出售重量': '出售重量',
        '出售单价': '出售单价',
        '出售价格': '出售价格'
    }
    available_sale_cols = [col for col in sale_mapping.keys() if col in df_sale.columns]
    df_sale_clean = df_sale[available_sale_cols].rename(columns=sale_mapping)
    if '资产 ID' in df_sale_clean.columns:
        df_sale_clean = df_sale_clean.drop_duplicates(subset=['资产 ID'], keep='last')
        df_main = pd.merge(df_main, df_sale_clean, on='资产 ID', how='left')

# ----------------------------------------------------
# 8. 聚合特征：死亡信息
# ----------------------------------------------------
if not df_dead.empty:
    print("正在整合死亡信息...")
    dead_mapping = {
        '资产ID': '资产 ID',
        '死亡日期': '死亡日期',
        '死亡原因': '死亡原因'
    }
    available_dead_cols = [col for col in dead_mapping.keys() if col in df_dead.columns]
    df_dead_clean = df_dead[available_dead_cols].rename(columns=dead_mapping)
    if '资产 ID' in df_dead_clean.columns:
        df_dead_clean = df_dead_clean.drop_duplicates(subset=['资产 ID'], keep='last')
        df_main = pd.merge(df_main, df_dead_clean, on='资产 ID', how='left')

# ----------------------------------------------------
# 9. 生命周期标签与养殖天数
# ----------------------------------------------------
print("正在计算生命周期标签与养殖天数...")
df_main['最终结局'] = '存栏'
if '死亡日期' in df_main.columns:
    df_main.loc[df_main['死亡日期'].notna(), '最终结局'] = '死亡'
if '出售日期' in df_main.columns:
    df_main.loc[df_main['出售日期'].notna(), '最终结局'] = '已出售'

if '出生日期' in df_main.columns:
    df_main['结束日期'] = pd.NaT
    if '死亡日期' in df_main.columns:
        df_main['结束日期'] = df_main['死亡日期'].fillna(df_main['结束日期'])
    if '出售日期' in df_main.columns:
        df_main['结束日期'] = df_main['出售日期'].fillna(df_main['结束日期'])

    df_main['出生日期'] = pd.to_datetime(df_main['出生日期'], errors='coerce')
    df_main['结束日期'] = pd.to_datetime(df_main['结束日期'], errors='coerce')

    mask = df_main['结束日期'].notna() & df_main['出生日期'].notna()
    df_main.loc[mask, '总生命周期(天)'] = (df_main.loc[mask, '结束日期'] - df_main.loc[mask, '出生日期']).dt.days

if '结束日期' in df_main.columns:
    df_main = df_main.drop(columns=['结束日期'])

# ----------------------------------------------------
# 10. 成本效益特征（直接回应“降低养殖成本”需求）
# ----------------------------------------------------
if '饲料总花费_元' in df_main.columns and '总增重' in df_main.columns:
    mask_cost = df_main['总增重'].notna() & (df_main['总增重'] > 0) & df_main['饲料总花费_元'].notna()
    df_main.loc[mask_cost, '每公斤增重成本'] = df_main.loc[mask_cost, '饲料总花费_元'] / df_main.loc[mask_cost, '总增重']

if '出售价格' in df_main.columns and '饲料总花费_元' in df_main.columns:
    mask_profit = df_main['出售价格'].notna() & df_main['饲料总花费_元'].notna()
    df_main.loc[mask_profit, '出栏盈亏'] = df_main.loc[mask_profit, '出售价格'] - df_main.loc[mask_profit, '饲料总花费_元']

if '饲料总花费_元' in df_main.columns and '喂养天数' in df_main.columns:
    mask_day = df_main['喂养天数'].notna() & (df_main['喂养天数'] > 0)
    df_main.loc[mask_day, '日均成本'] = df_main['饲料总花费_元'] / df_main['喂养天数']

# ----------------------------------------------------
# 11. 数据有效性与异常值标记
# ----------------------------------------------------
# 11.1 是否有有效生长记录
df_main['has_growth'] = df_main['平均日增重'].notna() & (df_main['成长记录次数'] > 0)

# 11.2 生长异常标记（仅针对有生长记录的牛）
df_main['is_adg_outlier'] = 0
if '平均日增重' in df_main.columns:
    # 日增重 <= 0 或 > 3 kg/天 标记为异常
    mask_adg = (df_main['平均日增重'].notna() &
                ((df_main['平均日增重'] <= 0) | (df_main['平均日增重'] > 3)))
    df_main.loc[mask_adg, 'is_adg_outlier'] = 1
if '总增重' in df_main.columns:
    # 总增重为负也标记异常
    df_main.loc[(df_main['总增重'].notna()) & (df_main['总增重'] < 0), 'is_adg_outlier'] = 1

# 将无生长记录的牛标记为异常（即不能用于生长分析）
df_main.loc[~df_main['has_growth'], 'is_adg_outlier'] = 1

# 11.3 可用于生长分析的样本量
growth_valid = df_main['has_growth'] & (df_main['is_adg_outlier'] == 0)
waterfall.append(('生长数据有效（去异常）', growth_valid.sum()))

# 11.4 可用于成本分析的样本量（有喂养记录 + 有总增重 + 总增重>0）
df_main['has_feed'] = df_main['喂养天数'].notna() & (df_main['喂养天数'] > 0)
df_main['has_cost'] = df_main['has_feed'] & df_main['总增重'].notna() & (df_main['总增重'] > 0)
waterfall.append(('饲喂+生长双全（成本分析样本）', df_main['has_cost'].sum()))

# ----------------------------------------------------
# 12. 样本筛选瀑布图输出
# ----------------------------------------------------
print("\n" + "="*60)
print("🌊 样本筛选瀑布图（每步保留样本量）")
print("-"*60)
max_width = 50
max_count = max([cnt for _, cnt in waterfall]) if waterfall else 1
for step_name, cnt in waterfall:
    bar_len = int(cnt / max_count * max_width) if max_count > 0 else 0
    diff_str = ""
    if len(waterfall) > 1:
        prev = waterfall[0][1]
        for s, c in waterfall:
            if s == step_name:
                break
            prev = c
        diff = cnt - prev
        diff_str = f"  ({diff:+d})" if diff != 0 else "  (-0)"
    print(f"  {step_name:<20} {cnt:>6}{diff_str}    ")
print("-"*60)

# ----------------------------------------------------
# 13. 缺失值诊断报告
# ----------------------------------------------------
print("\n" + "="*60)
print("📋 最终宽表缺失值报告")
print("-"*60)
missing = df_main.isnull().sum()
missing_pct = (missing / len(df_main) * 100).round(2)
missing_df = pd.DataFrame({'缺失数': missing, '缺失比例%': missing_pct})
missing_df = missing_df[missing_df['缺失数'] > 0].sort_values('缺失比例%', ascending=False)
if missing_df.empty:
    print("  ✅ 所有字段均无缺失值")
else:
    print(missing_df.to_string())
print("-"*60)

# ----------------------------------------------------
# 最终保存
# ----------------------------------------------------
print(f"\n✅ 最终分析宽表包含 {df_main.shape[1]} 个字段:")
print(list(df_main.columns))

output_path = os.path.join(DATA_DIR, "牛只全生命周期分析宽表.csv")
df_main.to_csv(output_path, index=False, encoding='utf-8-sig')

print("-" * 50)
print(f"核心分析宽表构建成功！共生成数据 {df_main.shape[0]} 行，{df_main.shape[1]} 个字段。")
print(f"输出路径: {output_path}")
print("\n数据预览前 5 行：")
print(df_main.head())