import pandas as pd
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
# 新增：喂养相关表
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
}

available_cols = [col for col in column_mapping.keys() if col in df_base.columns]
missing_cols = [col for col in column_mapping.keys() if col not in df_base.columns]
if missing_cols:
    print(f"\n[警告] 资产基础表中以下列不存在，将被跳过: {missing_cols}")

df_main = df_base[available_cols].rename(columns=column_mapping).drop_duplicates(subset=['资产 ID']).copy()

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

    # ---- 数据覆盖诊断 ----
    growth_unique_ids = set(df_growth['资产 ID'].unique())
    growth_matched = growth_unique_ids & main_ids_set
    print(f"\n📊 成长信息表: {len(df_growth)} 条记录, {len(growth_unique_ids)} 头牛")
    print(f"   资产ID范围: {min(growth_unique_ids)} ~ {max(growth_unique_ids)}")
    print(f"   匹配主表: {len(growth_matched)} 头, 未匹配: {len(growth_unique_ids) - len(growth_matched)} 头")
    if growth_matched:
        print(f"   匹配ID示例: {sorted(growth_matched)[:10]}")

    # 计算日增重
    df_growth = df_growth.sort_values(['资产 ID', '测量日期']).reset_index(drop=True)
    df_growth['上次体重'] = df_growth.groupby('资产 ID')['体重'].shift(1)
    df_growth['上次日期'] = df_growth.groupby('资产 ID')['测量日期'].shift(1)

    mask = df_growth['上次体重'].notna() & df_growth['上次日期'].notna()
    df_growth['日增重'] = 0.0
    df_growth.loc[mask, '日增重'] = (
        (df_growth.loc[mask, '体重'] - df_growth.loc[mask, '上次体重']) /
        (pd.to_datetime(df_growth.loc[mask, '测量日期']) - pd.to_datetime(df_growth.loc[mask, '上次日期'])).dt.days
    )

    growth_features = df_growth.groupby('资产 ID').agg(
        平均日增重=('日增重', 'mean'),
        最新体重=('体重', 'max'),
        成长记录次数=('资产 ID', 'count')
    ).reset_index()

    df_main = pd.merge(df_main, growth_features, on='资产 ID', how='left')

# ----------------------------------------------------
# 3. 聚合特征：用药与疾病风险
# ----------------------------------------------------
if not df_med.empty:
    df_med = df_med.rename(columns={'资产ID': '资产 ID'})

    med_unique_ids = set(df_med['资产 ID'].unique())
    med_matched = med_unique_ids & main_ids_set
    print(f"\n📊 用药信息表: {len(df_med)} 条记录, {len(med_unique_ids)} 头牛")
    print(f"   资产ID范围: {min(med_unique_ids)} ~ {max(med_unique_ids)}")
    print(f"   匹配主表: {len(med_matched)} 头, 未匹配: {len(med_unique_ids) - len(med_matched)} 头")

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

    vaccine_unique_ids = set(df_vaccine['资产 ID'].unique())
    vaccine_matched = vaccine_unique_ids & main_ids_set
    print(f"\n📊 防疫信息表: {len(df_vaccine)} 条记录, {len(vaccine_unique_ids)} 头牛")
    print(f"   资产ID范围: {min(vaccine_unique_ids)} ~ {max(vaccine_unique_ids)}")
    print(f"   匹配主表: {len(vaccine_matched)} 头, 未匹配: {len(vaccine_unique_ids) - len(vaccine_matched)} 头")

    vaccine_features = df_vaccine.groupby('资产 ID').agg(
        累计防疫次数=('资产 ID', 'count')
    ).reset_index()

    df_main = pd.merge(df_main, vaccine_features, on='资产 ID', how='left')
    df_main['累计防疫次数'] = df_main['累计防疫次数'].fillna(0)

# ----------------------------------------------------
# 5. 衍生特征：计算总增重
# ----------------------------------------------------
if '最新体重' in df_main.columns and '初始体重' in df_main.columns:
    df_main['总增重'] = df_main['最新体重'] - df_main['初始体重']

# ----------------------------------------------------
# 6. 聚合特征：饲料消耗（新增！按头×天汇总）
# ----------------------------------------------------
if not df_feed_detail.empty:
    print("正在整合饲料消耗信息...")
    # 统一列名映射
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

    # ---- 数据覆盖诊断 ----
    feed_unique_ids = set(df_feed['资产 ID'].unique())
    feed_matched = feed_unique_ids & main_ids_set
    print(f"\n📊 喂养记录明细表: {len(df_feed)} 条记录, {len(feed_unique_ids)} 头牛")
    print(f"   匹配主表: {len(feed_matched)} 头, 未匹配: {len(feed_unique_ids) - len(feed_matched)} 头")

    # 按资产 ID 汇总：总饲料量、总花费、喂养天数
    feed_features = df_feed.groupby('资产 ID').agg(
        精料总量_kg=('精料量_kg', 'sum'),
        草料总量_kg=('草料量_kg', 'sum'),
        酒糟总量_kg=('酒糟量_kg', 'sum'),
        饲料总花费_元=('饲料花费_元', 'sum'),
        喂养天数=('喂养日期', 'count'),
        日均精料_kg=('精料量_kg', 'mean'),
        日均草料_kg=('草料量_kg', 'mean'),
        日均酒糟_kg=('酒糟量_kg', 'mean'),
        日均花费_元=('饲料花费_元', 'mean'),
    ).reset_index()

    # 拼接到主表
    df_main = pd.merge(df_main, feed_features, on='资产 ID', how='left')

    # 计算饲料转化率（需要总增重数据）
    if '总增重' in df_main.columns and '精料总量_kg' in df_main.columns:
        # 精料转化率 = 总增重 ÷ 精料总量（越高越好）
        mask_fcr = df_main['精料总量_kg'].notna() & df_main['精料总量_kg'] > 0 & df_main['总增重'].notna()
        df_main.loc[mask_fcr, '精料转化率'] = (
            df_main.loc[mask_fcr, '总增重'] / df_main.loc[mask_fcr, '精料总量_kg']
        )
        # 总饲料转化率 = 总增重 ÷ (精料+草料+酒糟)
        df_main['总饲料_kg'] = df_main['精料总量_kg'].fillna(0) + df_main['草料总量_kg'].fillna(0) + df_main['酒糟总量_kg'].fillna(0)
        mask_fcr_total = df_main['总饲料_kg'] > 0 & df_main['总增重'].notna()
        df_main.loc[mask_fcr_total, '总饲料转化率'] = (
            df_main.loc[mask_fcr_total, '总增重'] / df_main.loc[mask_fcr_total, '总饲料_kg']
        )

    matched_count = df_main['喂养天数'].notna().sum()
    print(f"   实际合并成功: {matched_count} 头有喂养记录")

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
        '总金额': '总金额'
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
# 9. 衍生特征：生命周期标签与养殖天数
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

# 打印最终字段列表
print(f"\n✅ 最终分析宽表包含 {df_main.shape[1]} 个字段:")
print(list(df_main.columns))

# 保存最终的分析宽表
output_path = os.path.join(DATA_DIR, "牛只全生命周期分析宽表.csv")
df_main.to_csv(output_path, index=False, encoding='utf-8-sig')

print("-" * 50)
print(f"核心分析宽表构建成功！共生成数据 {df_main.shape[0]} 行，{df_main.shape[1]} 个字段。")
print(f"输出路径: {output_path}")
print("\n数据预览前 5 行：")
print(df_main.head())