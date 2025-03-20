import triton
import triton.language as tl
import torch

@triton.heuristics(
    {
        "BLOCK_M": lambda args: 128,
        "BLOCK_N": lambda args: 128, # 64 or 128
        "num_warps": lambda args: 8 if args["actual_seqlen_k"] < 250000 else 16,
        "num_stages": lambda args: 3, # for faster forward pass
    }
)
@triton.jit
def _einsum_topk_1_kernel(
        Q: tl.const,
        K: tl.const,
        Out,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qm: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_om: tl.constexpr,
        h_hk_ratio: tl.constexpr,
        actual_seqlen_k,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
):
    
    start_m = tl.program_id(1)
    off_b = tl.program_id(2)
    off_h = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = ((Q + off_b * stride_qb + off_h * stride_qh) +
              + stride_qm * (start_m * BLOCK_M + offs_m[:, None])
                + offs_d[None, :])
    # topk==1, no longer need an offs_d dimension
    out_ptrs = ((Out + off_b * stride_ob + off_h * stride_oh) +
               + stride_om * (start_m * BLOCK_M + offs_m))
    
    off_h_k = off_h // h_hk_ratio
    offs_n = tl.arange(0, BLOCK_N)
    # transpose here
    k_ptrs = ((K + off_b * stride_kb + off_h_k * stride_kh) +
               offs_n[None, :] * stride_kn + offs_d[:, None])
    
    
    # load q
    q = tl.load(q_ptrs)
    # max
    k_max = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    k_argmax = tl.zeros([BLOCK_M], dtype=tl.int32)

    # loop along seqlen_k
    for start_n in range(0, actual_seqlen_k, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # load k
        k = tl.load(k_ptrs + start_n * stride_kn, cache_modifier=".cg")
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, qk, allow_tf32=True, out_dtype=tl.float32)

        # max
        k_max_curr, k_argmax_curr = tl.max(qk, axis=1, return_indices=True) # keep_dim = false
        mask = k_max_curr >= k_max
        k_max = tl.where(mask, k_max_curr, k_max)
        k_argmax = tl.where(mask, (start_n + k_argmax_curr), k_argmax)

    o = k_argmax.to(Out.dtype.element_ty)
    tl.store(out_ptrs, o, cache_modifier=".cs")

    return

@triton.heuristics(
    {
        "BLOCK_M": lambda args: 128,
        # "BLOCK_N": lambda args: 256, # 64 or 128
        # "num_warps": lambda args: 8,
        "num_warps": lambda args: 8 if args["actual_seqlen_k"] < 250000 else 16,
        "num_stages": lambda args: 3, # for faster forward pass
    }
)
@triton.jit
def _einsum_local_max_kernel(
        Q: tl.const,
        K: tl.const,
        Out,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qm: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_on: tl.constexpr,
        h_hk_ratio: tl.constexpr,
        actual_seqlen_q,
        actual_seqlen_k,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
):
    
    start_n = tl.program_id(1)
    off_b = tl.program_id(2)
    off_h = tl.program_id(0)


    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # transpose here
    q_ptrs = ((Q + off_b * stride_qb + off_h * stride_qh) +
              + stride_qm * offs_m[None, :]
                + offs_d[:, None])
    
    offs_n = tl.arange(0, BLOCK_N)
    out_ptrs = ((Out + off_b * stride_ob + off_h * stride_oh + stride_on * start_n)
                 + offs_m)
    off_h_k = off_h // h_hk_ratio
    k_ptrs = ((K + off_b * stride_kb + off_h_k * stride_kh) +
                stride_kn * (start_n * BLOCK_N + offs_n)[:, None] + offs_d[None, :])
    
    
    # load q
    k = tl.load(k_ptrs)

    # loop along seqlen_q
    for start_m in range(0, actual_seqlen_q, BLOCK_M):
        # q = tl.load(q_ptrs, mask=offs_m < actual_seqlen_q - start_m)
        q = tl.load(q_ptrs + start_m * stride_qm, cache_modifier=".cg")
        
        kq_max = tl.max(tl.dot(k, q, allow_tf32=True, out_dtype=tl.float32), axis=0)
        o = kq_max.to(Out.dtype.element_ty)

        tl.store(out_ptrs + start_m, o, cache_modifier=".cs")

    return

def einsum_topk_func(q: torch.Tensor, k: torch.Tensor, topk: int) -> torch.Tensor:
    batch, num_heads, seqlen_q, d = q.shape
    batch_k, num_heads_k, seqlen_k, dk = k.shape
    assert q.stride(-1) == 1 and k.stride(-1) == 1
    assert num_heads % num_heads_k == 0, "num_heads must be divisible by num_heads_k"
    assert d == dk and batch_k == batch, "batch & head dimensions must match"
    assert q.dtype == k.dtype and q.dtype in [torch.float16, torch.bfloat16], "All tensors must have the same type. Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda, "All tensors must be sent to gpu"
    assert d == 128 and seqlen_q % 128 == 0 and seqlen_k % 128 == 0, f"Only support d == 128 && seqlen_q % 128 == 0 && seqlen_k % 128 == 0, but find d={d}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}"
    assert topk == 1, "Only support k == 1"


    if topk == 1:
        o = torch.empty((batch, num_heads, seqlen_q, topk), dtype=torch.int32, device=q.device)

        grid = lambda META: (num_heads, triton.cdiv(seqlen_q, META["BLOCK_M"]), batch)
        _einsum_topk_1_kernel[grid](
            q,
            k,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            num_heads // num_heads_k,
            seqlen_k,
            BLOCK_HEADDIM=d
        )
    else: # topk == 4
        BLOCK_N = 256
        o = torch.empty((batch, num_heads, triton.cdiv(seqlen_k, BLOCK_N), seqlen_q), dtype=q.dtype, device=q.device)

        grid = lambda META: (num_heads, triton.cdiv(seqlen_k, BLOCK_N), batch)
        _einsum_local_max_kernel[grid](
            q,
            k,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            num_heads // num_heads_k,
            seqlen_q,
            seqlen_k,
            BLOCK_N=BLOCK_N,
            BLOCK_HEADDIM=d,
        )
        o = (o.topk(topk, dim=-2, largest=True, sorted=False).indices).to(dtype=torch.int32)

    return o


def einsum_topk_unique_func(q: torch.Tensor, k: torch.Tensor, topk: int) -> torch.Tensor:
    batch, num_heads, seqlen_q, d = q.shape
    batch_k, num_heads_k, seqlen_k, dk = k.shape
    assert q.stride(-1) == 1 and k.stride(-1) == 1
    assert num_heads % num_heads_k == 0, "num_heads must be divisible by num_heads_k"
    assert d == dk and batch_k == batch, "batch & head dimensions must match"
    assert q.dtype == k.dtype and q.dtype in [torch.float16, torch.bfloat16], "All tensors must have the same type. Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda, "All tensors must be sent to gpu"
    assert d == 128 and seqlen_q % 128 == 0 and seqlen_k % 256 == 0, "Only support d == 128 && seqlen_q % 128 == 0 && seqlen_k % 256 == 0"
    assert topk == 1, "Only support k == 1"

    if topk == 1:
        o = torch.empty((batch, num_heads, seqlen_q, topk), dtype=torch.int32, device=q.device)

        grid = lambda META: (num_heads, triton.cdiv(seqlen_q, META["BLOCK_M"]), batch)
        _einsum_topk_1_kernel[grid](
            q,
            k,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            num_heads // num_heads_k,
            seqlen_k,
            BLOCK_HEADDIM=d
        )
    else: # topk == 4
        BLOCK_N = 256
        o = torch.empty((batch, num_heads, triton.cdiv(seqlen_k, BLOCK_N), seqlen_q), dtype=q.dtype, device=q.device)

        grid = lambda META: (num_heads, triton.cdiv(seqlen_k, BLOCK_N), batch)
        _einsum_local_max_kernel[grid](
            q,
            k,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            num_heads // num_heads_k,
            seqlen_q,
            BLOCK_N=BLOCK_N,
            BLOCK_HEADDIM=d,
        )
        o = o.topk(topk, dim=-2, largest=True, sorted=False).indices


    o = o.unique()
    return o