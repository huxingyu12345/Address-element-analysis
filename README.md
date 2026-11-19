# 中文地址要素解析

2025 天池极客挑战赛「闪购」地址要素解析。任务是把自由书写的中文地址拆成结构化要素（省、市、区县、道路、POI 等），按字符做序列标注。

数据源自 [CCKS2021 中文地址要素解析](https://tianchi.aliyun.com/dataset/109339)。方法是 **jieba 地址词典 + 手工特征 + BERT 句向量 + 线性链 CRF**。

## 环境

建议 Python 3.8+。在项目根目录执行：

```bash
pip install -r requirements.txt
```

BERT 句向量通过 ModelScope 加载 `tiansz/bert-base-chinese`，首次运行会下载模型。没有 GPU 会自动退回 CPU；也可以加 `--no-llm` 关掉语言模型特征。

GPU 设备号在 `address_ner/config.py` 的 `CUDA_DEVICE`（当前为 `cuda:7`），请按本机显卡改成 `cuda:0` 等。

## 快速开始

在项目根目录：

```bash
python main.py
```

默认完成：构建词典（若不存在）→ 训练 CRF → 对测试集预测。

没有 `data/final_test.txt` 时，会用验证集 `data/dev.txt` 作为预测对象，结果写到 `outputs/enhanced_dev_pred.txt`。

## 运行方式

统一入口是 `main.py`：

| 命令 | 作用 |
|---|---|
| `python main.py` 或 `python main.py all` | 训练 + 预测 |
| `python main.py train` | 只训练并保存模型 |
| `python main.py predict` | 加载已有模型做预测 |
| `python main.py dict` | 从训练集构建地址词典 |
| `python main.py dict --force` | 强制重建词典 |
| `python main.py --no-llm` | 全程关闭 BERT 特征 |
| `python main.py train --no-llm` | 无 BERT 训练 |

`python baseline.py` 仍可用，行为与 `python main.py` 相同。

## 数据

| 文件 | 说明 |
|---|---|
| `data/train_dev.txt` | 训练数据（训练集 + 验证集合并），CoNLL 字级标注 |
| `data/dev.txt` | 验证集，有金标 |
| `data/test.conll` | 测试集 CoNLL，标签为 `O` |
| `data/final_test.txt` | 官方测试原文（可选，`id\u0001文本`） |
| `address_dict.txt` | jieba 用户词典，可由 `python main.py dict` 生成 |
| `中文地址要素解析标注规范.pdf` | 官方标注规范 |

标注格式每行 `字符 标签`，空行分隔一条地址。标签为 **BIES**（Begin / Inside / End / Single），例如：

```text
浙 B-prov
江 I-prov
省 E-prov
杭 B-city
州 I-city
市 E-city
```

要素类型：`prov` `city` `district` `town` `community` `village_group` `devzone` `road` `roadno` `intersection` `poi` `subpoi` `houseno` `cellno` `floorno` `assist` `distance`，以及非地址内容 `O`。

## 输出

| 路径 | 说明 |
|---|---|
| `experiments/enhanced_model/enhanced_crf.pkl` | 训练得到的 CRF 模型 |
| `experiments/enhanced_model/feature_importance.txt` | 特征权重摘要 |
| `outputs/enhanced_pred.txt` | 官方测试集预测 |
| `outputs/enhanced_dev_pred.txt` | 用验证集代替测试时的预测 |

预测文件每行：

```text
ID<SOH>原文<SOH>标签序列
```

其中 `<SOH>` 为 `\u0001`。

## 项目结构

```text
.
├── main.py                 # 统一入口
├── baseline.py             # 兼容旧入口
├── requirements.txt
├── address_dict.txt
├── data/                   # 训练 / 验证 / 测试数据
├── outputs/                # 预测结果（运行后生成）
├── experiments/            # 模型与特征分析（运行后生成）
└── address_ner/
    ├── config.py           # 路径、超参数、标签表
    ├── dictionary.py       # 地址词典
    ├── dataset.py          # CoNLL 读写
    ├── embeddings.py       # BERT 句向量
    ├── features.py         # 字符 / 词 / LLM 特征
    ├── processor.py        # 数据与特征组装
    ├── model.py            # CRF
    └── pipeline.py         # 训练预测流程
```

改路径、正则化系数、迭代次数、是否使用 LLM，优先编辑 `address_ner/config.py`。
