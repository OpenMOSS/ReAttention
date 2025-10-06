<div align="center">
<h1>ReAttention: Training-Free Infinite Context with Finite Attention Scope</h1>

Xiaoran Liu<sup>1,2,3*</sup>, Ruixiao Li<sup>1,2*</sup>, Qipeng Guo<sup>2,3</sup>, Zhigeng Liu<sup>1</sup>, Yuerong Song<sup>1</sup>, 

Kai Lv<sup>1,3</sup>, Hang Yan<sup>3</sup>, Linlin Li<sup>4</sup>, Qun Liu<sup>4</sup>, Xipeng Qiu<sup>1,2†</sup>

<sup>1</sup> Fudan Univerisity, <sup>2</sup>Shanghai Innovation Institute, <sup>3</sup>Shanghai AI Laboratory, <sup>4</sup>Huawei Noah’s Ark Lab

[<a href="https://arxiv.org/abs/2407.15176">📝 Paper</a>] | [<a href="https://huggingface.co/papers/2407.15176">🤗 HF</a>] | [<a href="https://github.com/OpenMOSS/ReAttention">🚀 Code</a>]
</div>

## Introduction

In this work, we propose **ReAttention**, a training-free approach enabling LLM based on the self-attention mechanism to *break the maximum supported context length in length extrapolation* and *support an infinite context with a finite attention scope under sufficient memory resources*. It performs the position-agnostic top-k attention before the ordinary position-aware self-attention, freeing LLMs from the length extrapolation issue. 

We validate the performance of ReAttention on the LongBench, L-Eval, and InfiniteBench and demonstrate that it is on par with traditional methods. Furthermore, we also apply ReAttention on mainstream LLMs, including LLaMA3.1-8B and Mistral-v0.3-7B, enabling them to support context lengths of at least 1M and even expanding the context length of Qwen2-1.5B by 128× to 4M without any further training in Needle-In-A-Haystack. 

We also improve the efficiency of ReAttention with Triton and achieve an extrapolation without additional overhead. If you have questions about this work, please feel free to raise issues or send an email to xrliu24@m.fudan.edu.cn. If you find our paper useful, please consider citing the following paper:

```
@article{liu2024reattention,
  title={Reattention: Training-free infinite context with finite attention scope},
  author={Liu, Xiaoran and Li, Ruixiao and Guo, Qipeng and Liu, Zhigeng and Song, Yuerong and Lv, Kai and Yan, Hang and Li, Linlin and Liu, Qun and Qiu, Xipeng},
  journal={arXiv preprint arXiv:2407.15176},
  year={2024}
}
```

## Installation

### Prepare Your OpenCompass

We run our downstream evaluation based on [OpenCompass](https://github.com/open-compass/opencompass).

```bash
git clone https://github.com/open-compass/opencompass
cd opencompass
pip install -e .
```

The necessary Python packages we use and their corresponding versions.

```
flash-attn==2.7.4.post1
opencompass==0.4.2
torch==2.6.0
transformers==4.51.0
triton==3.2.0
```

### Prepare Your Model

Copy the folder `ReAttention/re_attention/` to `opencompass/models/` and add the following line to the end of `opencompass/models/__init__.py`.

```python
from .re_attention.re_attention_wrapper import ReAttentionCausalLM
```

## Evaluation

Copy the folder `ReAttention/eval/` to your OpenCompass directory and then you can try the following evaluations.

### Needle-In-A-Haystack Evaluation

1. Add a NIAH evaluation script with customizable context length and depth. Copy `ReAttention/needlebench/needlebench` to `opencompass/configs/datasets/needlebench` and replace `opencompass/configs/summarizers/needlebench.py` with `ReAttention/needlebench/needlebench.py`.

2. Edit the prompt format of the RULER benchmark to enable the base model to respond more effectively by replacing `opencompass/datasets/needlebench/origin.py` with `ReAttention/needlebench/origin.py`.

3. You can also modify the plotting code in `opencompass/summarizers/needlebench.py` as shown in `ReAttention/needlebench/needlebench_summarizer.py`, which is optional.

4. Execute the following command.

```bash
python run.py eval/eval_reattn_niah.py --dump-eval-details -r
```

### Long-Context Benchmark Evaluation

1. Execute the following command for LongBench and LEval.

```bash
python run.py eval/eval_reattn_long.py --dump-eval-details -r
```

1. Execute the following command for InfiniteBench (EnMC, EnQA, EnSum).

```bash
python run.py eval/eval_reattn_infinite.py --dump-eval-details -r
```

For more evaluation details, please refer to branch `main`.

## News

- [2025.03.20] 🎉⚡🎉 We release our Triton code and evaluation code forked from [OpenCompass](https://github.com/open-compass/OpenCompass/) on Github.
- [2025.03.19] 🎉🚀🎉 We release the third version of the paper on arXiv. 
- [2025.01.22] 🎉🎉🎉 Our paper is accepted by ICLR 2025. See in [OpenReview](https://openreview.net/forum?id=KDGP8yAz5b).
- [2024.10.05] 🎉🚀🎉 We release the second version of the paper on arXiv, [ReAttention: Training-Free Infinite Context with Finite Attention Scope](https://arxiv.org/abs/2407.15176v2). We add Triton-based optimization in this version and reorganize our work.
- [2024.07.21] 🎉🚀🎉 We release the first version of the paper on arXiv, [Farewell to Length Extrapolation, a Training-Free Infinite Context with Finite Attention Scope](https://arxiv.org/abs/2407.15176v1). 

## Todo

- [ ] Realize the precise Triton kernel for top-k attention.
- [ ] Release the code for efficiency analysis.
- [x] Organize the evaluation code and give necessary comments.
- [x] Release the approximate Triton kernel for top-k attention reported in our paper.
- [x] Release the code for long-context evaluation.