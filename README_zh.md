# POLIS-Bench 项目说明

<p align="center">
  <img src="https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-EN%20%7C%20CN-blue" alt="Languages">
  <img src="https://img.shields.io/badge/%E4%BB%BB%E5%8A%A1-%E6%9D%A1%E6%AC%BE%E6%A3%80%E7%B4%A2%20%7C%20%E7%AD%96%E7%95%A5%E7%94%9F%E6%88%90%20%7C%20%E5%90%88%E8%A7%84%E5%88%A4%E5%AE%9A-orange" alt="Tasks">
  <img src="https://img.shields.io/badge/python-3.10+-brightgreen" alt="Python">
  <img src="https://img.shields.io/badge/license-Research%20Use-lightgrey" alt="License">
</p>

> 多语言政策问答基准，提供从数据处理、模型微调到推理评测、可视化的完整开源流水线。

## 目录
- [项目概览](#项目概览)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
  - [1. 环境准备](#1-环境准备)
  - [2. 数据处理](#2-数据处理)
  - [3. 模型微调](#3-模型微调)
  - [4. 推理与评测](#4-推理与评测)
- [命令速查表](#命令速查表)
- [可视化](#可视化)
- [数据与隐私](#数据与隐私)
- [贡献指南](#贡献指南)
- [许可](#许可)

## 项目概览
- 覆盖中英双语政策场景，任务包含条款检索、方案生成与合规判定。
- 每个流程步骤均配备可复用脚本，可在本仓库内完成数据准备、LoRA 微调、推理评测与绘图。
- 模块化目录拆分，便于定制模型或集成现有基础设施。

## 目录结构

| 路径 | 说明 |
| ---- | ---- |
| `configs/` | DeepSeek-R1、Qwen3 LoRA 配置示例（兼容 LLaMA-Factory）。 |
| `datasets/` | 脚本生成的派生数据输出目录（过滤结果、切分文件、Prompt 后数据等）。 |
| `evaluation/` | 语义相似度、LLM Judge、统计汇总等评测脚本。 |
| `finetuning/` | 整合的一体化数据预处理脚本。 |
| `inference/` | 面向 OpenAI 兼容接口的批量推理脚本。 |
| `POLIS_dataset/` | 官方训练/测试集以及蒸馏语料。 |
| `visualization/` | 论文级可视化脚本（雷达图等）。 |

> **提示**：原始政策文本不随仓库发布，请按版权要求安全存储，并使用仓库脚本生成所需衍生文件。

## 快速开始

### 1. 环境准备

```bash
conda create -n polis-bench python=3.10 -y
conda activate polis-bench
pip install -r requirements.txt  # 请根据下列依赖自行整理
```

主要依赖：`tqdm`、`numpy`、`pandas`、`sentence-transformers`、`scikit-learn`、`matplotlib`、`openai (>=1.0)`。

### 2. 数据处理

使用 `finetuning/prepare_dataset.py` 完成过滤、切分与 Prompt 构建：

```bash
# 过滤原始模型输出，去除 model/answer 字段
python finetuning/prepare_dataset.py filter \
  --input /path/to/raw_generations.jsonl \
  --output datasets/filtered_generations.jsonl

# 按设定种子拆分训练/测试集
python finetuning/prepare_dataset.py split \
  --input datasets/filtered_generations.jsonl \
  --train-output datasets/train.jsonl \
  --test-output datasets/test.jsonl \
  --train-size 2558 --seed 42

# 为训练样本拼接指令式 Prompt
python finetuning/prepare_dataset.py format \
  --input datasets/train.jsonl \
  --output datasets/train_prompted.jsonl
```

如需自定义 Prompt，可使用 `--template` 参数替换默认模板。

### 3. 模型微调

`configs/` 中的配置可直接配合 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 使用，记得替换模型路径与数据路径：

```bash
llama_factory-cli train configs/ds_llama3_lora_sft.yaml
```

训练输出默认写入 `./outputs/`。

### 4. 推理与评测

利用 OpenAI 兼容接口批量推理：

```bash
python inference/run_inference.py \
  --input POLIS_dataset/test_set/test.jsonl \
  --output results/deepseek_predictions.jsonl \
  --failed results/deepseek_failed.jsonl \
  --model deepseek-r1-distill-llama-8b \
  --base-url http://localhost:8000/v1 \
  --api-key YOUR_KEY
```

随后对结果进行打分与统计：

```bash
# 语义相似度得分
python evaluation/semantic_similarity.py \
  --input results/deepseek_predictions.jsonl \
  --output results/semantic_scores

# LLM Judge
python evaluation/llm_judge.py \
  --input results/semantic_scores \
  --output results/llm_judge \
  --failed results/llm_judge_failed \
  --base-url https://api.example.com/v1 \
  --api-key sk-*** \
  --model my-judge-model

# 汇总统计输出 CSV
python evaluation/aggregate_results.py \
  --input results/llm_judge \
  --output results/aggregated
```

## 命令速查表

| 阶段 | 命令 | 用途 |
| ---- | ---- | ---- |
| 数据过滤 | `prepare_dataset.py filter` | 去除无用字段，得到干净语料。 |
| 数据切分 | `prepare_dataset.py split` | 以固定种子划分训练/测试集。 |
| Prompt 构建 | `prepare_dataset.py format` | 为样本附加 SFT 指令模板。 |
| 微调 | `llama_factory-cli train ...` | 使用 LLaMA-Factory 运行 LoRA 训练。 |
| 推理 | `run_inference.py` | 将样本批量发送至 OpenAI 兼容接口。 |
| 语义评估 | `semantic_similarity.py` | 多语言语义相似度计算。 |
| LLM Judge | `llm_judge.py` | 输出 `[Correct]/[Incorrect]` 标签。 |
| 汇总统计 | `aggregate_results.py` | 按模型/语言/任务输出统计报告。 |
| 可视化 | `visualization/model_comparison.py` | 重现论文中的雷达图。 |

## 可视化

运行 `visualization/model_comparison.py` 可生成论文中的雷达图，保存至 `visualization/model_comparison.png`。

## 数据与隐私
- `POLIS_dataset/` 提供蒸馏语料及官方训练/测试划分。
- 原始政策文本与 Prompt 保持私有，请使用仓库脚本生成所需衍生文件。

## 贡献指南
欢迎通过 Issue 或 PR 提交 Bug 修复、功能扩展或文档改进。新增脚本时请遵守目录规范，避免上传受限政策原文。

## 许可
本仓库的开放数据仅限科研用途。如需二次分发派生内容，请先与维护者联系。


