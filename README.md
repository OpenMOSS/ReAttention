# ReAttention: Training-Free Infinite Context with Finite Attention Scope

<div align="center">
 <p align="center">

   <a href="https://arxiv.org/abs/2407.15176">📝 Paper</a> | <a href="https://huggingface.co/papers/2407.15176">🤗HF</a>

</p>
</div>

> In this work, we propose **ReAttention**, a training-free approach enabling LLM based on the self-attention mechanism to *break the maximum supported context length in length extrapolation* and *support an infinite context with a finite attention scope under sufficient memory resources*. It performs the position-agnostic top-k attention before the ordinary position-aware self-attention, freeing LLMs from the length extrapolation issue. 
> 
> We validate the performance of ReAttention on the LongBench, L-Eval, and InfiniteBench and demonstrate that it is on par with traditional methods. Furthermore, we also apply ReAttention on mainstream LLMs, including LLaMA3.1-8B and Mistral-v0.3-7B, enabling them to support context lengths of at least 1M and even expanding the context length of Qwen2-1.5B by 128× to 4M without any further training in Needle-In-A-Haystack. 
>
> We also improve the efficiency of ReAttention with Triton and achieve an extrapolation without additional overhead. If you have questions about this work, please feel free to raise issues or send an email to xrliu24@m.fudan.edu.cn. If you find our paper useful, please consider citing the following paper:

```bibtex
@misc{liu2025spakelongcontextlargelanguage,
    title={ReAttention: Training-Free Infinite Context with Finite Attention Scope},
    author={Liu, Xiaoran and Li, Ruixiao and Guo, Qipeng and Liu, Zhigeng and Song, Yuerong and Lv, Kai and Yan, Hang and Li, Linlin and Liu, Qun and Qiu, Xipeng},
    year={2024},
    eprint={2407.15176},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2407.15176}, 
}
```

## News

- [2025.03.20] 🎉⚡🎉 We release our Triton code and evaluation code forked from [OpenCompass](https://github.com/open-compass/OpenCompass/) on Github.
- [2025.03.19] 🎉🚀🎉 We release the third version of the paper on arXiv. 
- [2025.01.22] 🎉🎉🎉 Our paper is accepted by ICLR 2025. See in [OpenReview](https://openreview.net/forum?id=KDGP8yAz5b).
- [2024.10.05] 🎉🚀🎉 We release the second version of the paper on arXiv, [ReAttention: Training-Free Infinite Context with Finite Attention Scope](https://arxiv.org/abs/2407.15176v2). We add Triton-based optimization in this version and reorganize our work.
- [2024.07.21] 🎉🚀🎉 We release the first version of the paper on arXiv, [Farewell to Length Extrapolation, a Training-Free Infinite Context with Finite Attention Scope](https://arxiv.org/abs/2407.15176v1). 

## Todo

- [ ] Realize the precise Triton kernel for top-k attention.
- [ ] Release the code for efficiency analysis.
- [ ] Organize the evaluation code and give necessary comments.
- [x] Release the approximate Triton kernel for top-k attention reported in our paper.
- [x] Release the code for long-context evaluation.
