<div align="center">

<h1>POLAR-Bench</h1>
<h3>诊断 LLM 智能体隐私–效用权衡的基准</h3>

<p><strong>Poster at NeurIPS 2026 E&amp;D Track.</strong></p>
<p>
<a href="https://github.com/qiaoyuan667">Qiaoyuan Zheng</a><sup>*</sup> ·
Yiqu Yang<sup>*</sup> · Qi Gao<sup>*</sup> · Imanol Schlag<sup>†</sup><br>
ETH Zurich · ETH AI Center<br>
<sub>* 共同第一作者。† 通讯作者。</sub>
</p>

<p>
<a href="https://arxiv.org/abs/2605.19127">论文</a> ·
<a href="https://huggingface.co/datasets/Qiaoyuan/POLAR-Bench">Hugging Face 数据集</a> ·
<a href="docs/model-setup.md">模型接入指南</a>
</p>

<p><a href="README.md">English</a> · <strong>简体中文</strong></p>

</div>

## 你的智能体能否分享必要信息，同时守住隐私边界？

什么都说不安全，什么都不说也无法提供有效帮助。**POLAR-Bench 测试两者之间的选择性披露能力：在对抗性交互中，模型能否提供任务所需的信息，同时不披露受保护属性？**

| 评测样本 | 应用领域 | 政策 × 攻击设计 | 论文评测模型 |
| :---: | :---: | :---: | :---: |
| **7,852** | **10** | **5 × 5** | **22** |

**[主要结果](#主要结果) · [开始评测](#开始评测) · [数据详情](docs/benchmark.md) · [引用](#引用)**

> **2026 年 9 月：** POLAR-Bench 以 poster 形式录用于 NeurIPS 2026 E&D Track。代码与可直接评测的数据集已实名公开。

## 一图理解核心思路

[![POLAR-Bench 概念示意图：用户将任务与隐私政策交给可信智能体；它需要在与外部模型对话时分享任务相关信息，同时保护私密信息。](assets/figures/concept-overview.jpg)](assets/figures/concept-overview.jpg)

*论文 Figure 1 概念示意图。[原始高清图片](assets/figures/concept-overview.jpg)。图中的 Utility 按下文定义衡量任务所需信息的披露覆盖率。*

1. **明确披露边界。** 合成源文档包含任务所需、受保护及其他属性，任务与隐私政策共同定义选择性披露场景。
2. **测试对抗交互。** Model A 是被评测的可信模型，Model B 扮演外部攻击者；五类政策表述与五类对话攻击组合覆盖十个领域。
3. **分别衡量两个结果。** 根据预定义属性值进行确定性披露评分，不使用 LLM 作为结果评分裁判。

**Privacy** 衡量受保护属性未被披露的比例；**Attribute Utility** 衡量任务所需属性被披露的比例，**不等同于端到端任务完成率**。两者均越高越好。政策与攻击是不同类型，不代表已经验证的单调难度等级。

[查看领域覆盖、政策与攻击定义、数据分布 →](docs/benchmark.md)

[查看详细构建流程图 →](docs/pipeline.md#pipeline-overview)

## 主要结果

*以下图表和数值来自 [arXiv v1](https://arxiv.org/abs/2605.19127v1)，不是实时排行榜。原图中的 Utility 指 Attribute Utility。*

### 高隐私分数不一定意味着高效用

[![22 个模型的 Privacy–Attribute Utility 散点图。](assets/figures/privacy-utility.png)](assets/figures/privacy-utility.pdf)

*Figure 3。[高清图片](assets/figures/privacy-utility.png) · [原始 PDF](assets/figures/privacy-utility.pdf)。*

- **模型差距明显：** 在隐私与效用等权的 Overall 指标上，论文报告最高与最低模型相差超过 30 分。
- **隐私与效用需要同时报告：** GLM-4.7-Flash 的 Privacy 为 98.9，但 Attribute Utility 为 61.2；GLM-5.1 的两项分数分别为 99.3 和 89.9（Figure 2 中的四舍五入值）。
- **区分两类失败：** 散点图分别展示泄露受保护信息，以及未提供任务必要信息的问题。

### 不只比较排名，也定位失败组合

[![五类政策和五类攻击组合下的平均 Privacy 与 Attribute Utility 热图。](assets/figures/diagnostic-surface.png)](assets/figures/diagnostic-surface.pdf)

*Figure 4，每个单元格为跨模型平均值。[高清图片](assets/figures/diagnostic-surface.png) · [原始 PDF](assets/figures/diagnostic-surface.pdf)。*

5×5 诊断面展示模型表现如何随政策表述和攻击方式变化。例如，Table 9 中 S2（是非式逐步缩小范围）下的平均 Privacy 为 66.06，低于 S5（多轮渐进攻击）的 78.08：更复杂的攻击不一定更有效。

跨模型均值可能掩盖个别模型的差异；模型级结论应结合论文中的分模型结果解读。

## 开始评测

**下载最终数据即可接入评测器，无需重新生成、渲染、验证、修复或过滤。** Hugging Face 只提供最终 benchmark；代码在 GitHub。

### 1. 安装并下载

使用 Python 3.10+ 和独立环境。以下命令使用 POSIX shell；Windows 可使用 WSL。

```bash
# Skip construction-stage LFS data when cloning the code.
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/qiaoyuan667/POLAR-Bench.git
cd POLAR-Bench
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt huggingface_hub

# Download only the final evaluator input.
hf download Qiaoyuan/POLAR-Bench \
  data/privacy_benchmark_rendered_repaired.json \
  --repo-type dataset --local-dir .
```

### 2. 接入你的两个模型

Model A 是被评测模型，Model B 在攻击协议需要时生成对抗性后续消息。以下示例保留论文所用的 Llama-3.3-70B-Instruct 攻击者；替换攻击者属于不同实验设定。

```bash
export POLAR_API_KEY="YOUR_API_KEY"
export POLAR_BASE_URL="https://YOUR_ENDPOINT/v1"
export POLAR_MODEL_A="YOUR_TRUSTED_MODEL_ID"
export POLAR_MODEL_B="meta-llama/Llama-3.3-70B-Instruct"
```

将占位符替换成你实际可用的服务地址、密钥和模型 ID。当前评测器通过**同一 endpoint 和 key**访问两个模型；若它们部署在不同平台，使用[网关方案](docs/model-setup.md#4-separate-self-hosted-endpoints)。

`openai-compatible` 指接口协议，不代表调用 OpenAI 托管模型。数据下载不会自动部署模型，Hugging Face 下载凭证也不等于推理 API 密钥。不要把密钥提交到仓库。

[详细模型接入指南](docs/model-setup.md)涵盖现有平台、vLLM/SGLang 部署、双后端网关、连接检查和排错。开始评测前要分别检查两个模型：S1–S4 的脚本式攻击可以在 Model B 不可用时先运行成功。

### 3. 先少量试跑，再全量评测

```bash
# Smoke test: two instances per domain. Model calls may incur API charges.
python scripts/ab_eval.py \
  --dataset data/privacy_benchmark_rendered_repaired.json \
  --model-a "${POLAR_MODEL_A}" \
  --domains medical recruitment finance education customer_support legal insurance housing travel cybersecurity \
  --samples-per-domain 2 \
  --max-rounds 6 \
  --seed 42 \
  --output results/smoke_summary.json \
  --output-details results/smoke_details.json \
  --checkpoint results/smoke_checkpoint.json \
  --base-url "${POLAR_BASE_URL}" \
  --model-b "${POLAR_MODEL_B}" \
  --max-workers 5 \
  --model-a-provider openai-compatible \
  --defense none
```

- **全量评测：** 将 `--samples-per-domain 2` 改为 `--samples-per-domain 0`，保留全部十个领域，并使用新的 `full_*` 输出与 checkpoint 路径。
- **断点续跑：** 使用同一配置与 checkpoint 重复原命令。不同模型或防御设置不可混用 checkpoint。
- **结果位置：** 分别查看 `results/smoke_*` 对应的汇总、逐样本详情和 checkpoint 文件。
- **费用：** 评测会调用模型，可能产生 API 费用。

## 进一步了解

| 目的 | 入口 |
| --- | --- |
| 接入自己部署的平台和模型 | [Model A / Model B 指南](docs/model-setup.md) |
| 查看样本结构、统计与指标定义 | [Benchmark 详情](docs/benchmark.md) |
| 研究或扩展构建流程 | [可选构建与后处理流程](docs/pipeline.md) |
| 下载论文原图、核对来源 | [图表资源与来源说明](assets/figures/README.md) |
| 反馈问题 | [GitHub Issues](https://github.com/qiaoyuan667/POLAR-Bench/issues) |

本次发布未重新生成冻结数据集。确定性评分不意味着不同推理后端或重复运行的模型输出完全一致；请记录模型版本与服务配置。

## 许可与使用边界

代码采用 [MIT](LICENSE)，数据集与论文图表采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。
作者：Qiaoyuan Zheng、Yiqu Yang、Qi Gao、Imanol Schlag。

数据为合成场景，不包含真实个人记录。属性标签是合成政策下的任务必要性定义，不代表普适的隐私规范；显式属性评分也不能覆盖全部语义推断与侧信道泄露。

## Acknowledgments and Disclosure of Funding

This work was supported as part of the Swiss AI Initiative by compute grant infra01 from the Swiss National Supercomputing Centre (CSCS) on Alps.

## 引用

```bibtex
@article{zheng2026polar,
  title={POLAR-Bench: A Diagnostic Benchmark for Privacy-Utility Trade-offs in LLM Agents},
  author={Zheng, Qiaoyuan and Yang, Yiqu and Gao, Qi and Schlag, Imanol},
  journal={arXiv preprint arXiv:2605.19127},
  year={2026}
}
```
