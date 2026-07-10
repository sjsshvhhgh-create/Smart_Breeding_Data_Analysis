# 数字化赋能智慧养殖 — 数据挖掘项目

> **课题来源**：金惠家科技智慧牧养平台  
> **核心目标**：利用真实牛场运维数据，通过多方法数据挖掘，回答“如何降低养殖成本与风险”。

## 📖 项目简介

本项目针对肉牛养殖中 **单体成本高、养殖周期长、牛犊选择难、饲喂不科学、防疫不及时** 五大痛点，基于 **金惠家智慧牧养平台** 提供的约 2,657 头牛的全生命周期数据，进行了完整的数据预处理、特征工程与多方法建模分析。

通过**多元回归、贝叶斯估计、聚类分析**三种方法的交叉验证，得到核心结论：

> **在 8 月龄左右的犊牛中，初始体重是最可靠的生长预测指标。优先选择初始体重 ≥ 230 kg 的犊牛，预期日增重可提高约 17.8%，出栏率提高近 4 倍。**

## 🗂️ 项目结构

```
.
├── src/                          # 所有源代码
│   ├── data_processor.py         # ① 原始 Excel → CSV 清洗
│   ├── build_features.py         # ② 15 表 → 1 张宽表 + 特征工程
│   ├── data_split.py             # ③ 60/20/20 分层三集划分
│   ├── data_analysis.py          # ④ 描述性统计 + 可视化
│   ├── mining_regression.py      # ⑤ 多元回归：寻找促长因子
│   ├── mining_bayesian.py        # ⑥ 贝叶斯：量化不确定性
│   ├── mining_clustering.py      # ⑦ 聚类：牛犊科学分群
│   └── summary_report.py         # ⑧ 三方法交叉验证汇总
│
├── data/                         # 数据文件夹（不在仓库中，需本地生成）
│   ├── raw/                      # ← 将原始 Excel 文件放这里
│   ├── processed/                # 自动生成：清洗后 CSV + 宽表
│   ├── split/                    # 自动生成：train/val/test 三集
│   └── figures/                  # 自动生成：所有分析图表
│
├── docs/                         # 项目文档
│   ├── 目标.txt                  # 课题需求
│   ├── 数据信息.txt              # 原始数据说明
│   ├── 分析挖掘.md               # 分析纲要
│   └── 报告.md                   # 最终分析报告
│
├── requirements.txt              # Python 依赖库
├── .gitignore                    # Git 忽略规则
└── README.md                     # 本文件
```

## 🚀 快速复现指南

### 1. 环境准备
- Python 3.8+（推荐 3.9 或 3.10）
- Windows / macOS / Linux 均可

**克隆仓库**
```bash
git clone <仓库地址>
cd <项目目录>
```

**创建虚拟环境（推荐）**
```bash
python -m venv .venv
# Windows 激活
.venv\Scripts\activate
# macOS / Linux 激活
source .venv/bin/activate
```

**安装依赖**
```bash
pip install -r requirements.txt
```

### 2. 准备原始数据
将 15 张原始 Excel 数据表放入 `data/raw/` 目录。  
表名应与代码中读取的名称一致（例如 `资产基础表.xlsx`、`成长信息表.xlsx` 等）。

### 3. 按顺序运行脚本
**必须严格按照顺序执行，每一步生成的结果是下一步的输入。**

| 步骤 | 命令 | 功能 |
|------|------|------|
| ① | `python src/data_processor.py` | 清洗原始 Excel → 输出 CSV 至 `data/processed/` |
| ② | `python src/build_features.py` | 多表关联 → 构建宽表 `牛只全生命周期分析宽表.csv` |
| ③ | `python src/data_split.py` | 划分训练/验证/测试集至 `data/split/` |
| ④ | `python src/data_analysis.py` | 探索性分析，图表输出至 `figures/data_analysis/` |
| ⑤ | `python src/mining_regression.py` | 多元线性回归 |
| ⑥ | `python src/mining_bayesian.py` | 贝叶斯参数估计 |
| ⑦ | `python src/mining_clustering.py` | K-Means 聚类分析 |
| ⑧ | `python src/summary_report.py` | 三方法交叉验证汇总 |

### 4. 查看结果
- **数据输出**：`data/processed/`、`data/split/`
- **分析图表**：`figures/data_analysis/`、`figures/regression/`、`figures/bayesian/`、`figures/clustering/`、`figures/summary/`
- **最终报告**：`docs/报告.md`

## 📊 分析方法概览

| 方法 | 目标 | 核心结论 |
|------|------|---------|
| 多元线性回归 | 识别日增重关键因子 | 初始体重是唯一强显著正向因子 |
| 贝叶斯估计 | 量化系数不确定性 | 初始体重后验区间不含 0，结论稳健 |
| K-Means 聚类 | 犊牛科学分群 | 8 月龄、体重 ≥ 230 kg 为优质群体 |
| Bootstrap 交叉验证 | 多方法互锁 | 三种方法结论完全一致 |

## ⚠️ 注意事项
- 原始数据未包含在仓库中（已在 `.gitignore` 中忽略），请自行获取后放入 `data/raw/`。
- 所有处理后的数据和图表均可在本地运行脚本重新生成。
- 本分析使用的样本存在数据断层（生长记录覆盖率 19%，饲喂记录覆盖率 4%），报告中对各结论均标注了适用样本量，请勿过度外推。

## 📝 相关文档
- [数据挖掘纲要](docs/分析挖掘.md)
- [最终分析报告](docs/报告.md)
- [课题需求说明](docs/目标.txt)
```
