import torch
from typing import Union, Tuple
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

class RotaryEmbeddingESM(torch.nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(
        self, 
        # dim: int, 
        # base: Union[int, float] = 10000,
        # distance_scale: Union[int, float] = 1,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config=None,
    ):
        super().__init__()
        # self.base = base
        # self.distance_scale = distance_scale

        # inv_freq = 1.0 / (
        #     base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float32) / dim)
        # )
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        
        self.distance_scale = 1
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = -1
        self._cos_cached = None
        self._sin_cached = None

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, x, length, right, cos, sin):
        dtype = x.dtype
        if cos.dim() == 2:
            cos = cos[right-length:right, :]
            sin = sin[right-length:right, :]
        elif cos.dim() == 3:
            cos = cos[:, right-length:right, :]
            sin = sin[:, right-length:right, :]
        elif  cos.dim() == 4:
            cos = cos[:, :, right-length:right, :]
            sin = sin[:, :, right-length:right, :]
            
        return ((x.float() * cos.to(x.device)) + (self.rotate_half(x).float() * sin.to(x.device))).to(dtype)

    def _update_cos_sin_tables(self, x, seq_dim):
        seq_len = x.size(seq_dim)
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device = x.device).type_as(self.inv_freq)
            freqs = torch.outer(t * self.distance_scale, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim = -1)
            if x.dim() == 2:
                self._cos_cached = emb.cos()
                self._sin_cached = emb.sin()
            elif x.dim() == 3:
                self._cos_cached = emb.cos()[None, :, :]
                self._sin_cached = emb.sin()[None, :, :]
            elif x.dim() == 4:
                self._cos_cached = emb.cos()[None, None, :, :]
                self._sin_cached = emb.sin()[None, None, :, :]
        return self._cos_cached, self._sin_cached

    def _update_cos_sin_tables_len(self, seq_len, device, dim = None):
        if seq_len > self._seq_len_cached:
            if dim is None:
                assert self._cos_cached is not None
                dim = self._cos_cached.dim()

            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
            freqs = torch.outer(t * self.distance_scale, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim = -1)
            if dim == 2:
                self._cos_cached = emb.cos()
                self._sin_cached = emb.sin()
            elif dim == 3:
                self._cos_cached = emb.cos()[None, :, :]
                self._sin_cached = emb.sin()[None, :, :]
            elif dim == 4:
                self._cos_cached = emb.cos()[None, None, :, :]
                self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def apply_rotary_pos_emb_one_angle(
        self, x: torch.Tensor, index
    ):
        dtype = x.dtype
        cos, sin = self._update_cos_sin_tables_len(index, x.device)
        if cos.dim() == 2:
            cos = cos[index-1:index, :]
            sin = sin[index-1:index, :]
        elif cos.dim() == 3:
            cos = cos[:, index-1:index, :]
            sin = sin[:, index-1:index, :]
        elif  cos.dim() == 4:
            cos = cos[:, :, index-1:index, :]
            sin = sin[:, :, index-1:index, :]
            
        return ((x.float() * cos.to(x.device)) + (self.rotate_half(x).float() * sin.to(x.device))).to(dtype)


    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_dim= -2) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dim=seq_dim)
        return (
            self.apply_rotary_pos_emb(q, q.size(seq_dim), k.size(seq_dim), self._cos_cached, self._sin_cached),
            self.apply_rotary_pos_emb(k, k.size(seq_dim), k.size(seq_dim), self._cos_cached, self._sin_cached),
        )
