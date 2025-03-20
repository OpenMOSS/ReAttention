import math
import copy
import importlib.metadata
import json
import os
import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import einops
from packaging import version

from torch import nn

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import is_hqq_available, is_quanto_available, logging

# from .einsum_topk_fuse202408212109 import einsum_topk_func
from .einsum_topk_fuse20240907 import einsum_topk_func

if is_quanto_available():
    quanto_version = version.parse(importlib.metadata.version("quanto"))
    if quanto_version >= version.parse("0.2.0"):
        from quanto import AffineQuantizer, MaxOptimizer, qint2, qint4

if is_hqq_available():
    from hqq.core.quantize import Quantizer as HQQQuantizer

logger = logging.get_logger(__name__)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:24'


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings


    ################# begin # remove input_ids #################
    @torch.no_grad()
    def forward(self, x, position_ids=None):
        if position_ids == None:
            seq_len = x.shape[2]
            position_ids = torch.arange(seq_len)[None, :].float().to(device=x.device)
            
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(x.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float().to(device=inv_freq_expanded.device)
        # position_ids_expanded = torch.arange(seq_len)[None, None, :].float().to(device=inv_freq_expanded.device)
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    ################## end ## remove input_ids #################


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    ################# begin # remove input_ids #################
    def forward(self, x, position_ids=None):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        if position_ids == None:
            seq_len = x.shape[2]
            position_ids = torch.arange(seq_len)[None, :].float().to(device=x.device)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(x.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float().to(device=inv_freq_expanded.device) / self.scaling_factor
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    ################## end ## remove input_ids #################


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    ################# begin # remove input_ids #################
    def forward(self, x, position_ids=None):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = x.shape[2]
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin
    ################## end ## remove input_ids #################


class LlamaScalingNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with NTK scaling based on https://arxiv.org/abs/2310.05209. """

    ################# begin # remove input_ids #################
    def forward(self, x, position_ids=None):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = x.shape[2] if position_ids == None else int(position_ids.max())
        if seq_len > self.max_position_embeddings:
            # order = math.ceil(math.log(seq_len / (2 * math.pi), self.max_position_embeddings / (2 * math.pi)))
            # base = (self.base ** order) ** (self.dim / (self.dim - 2))
            dim_c = math.log(self.max_position_embeddings / (2 * math.pi), self.base)
            dim_c = 2 * math.ceil(self.dim * dim_c / 2)
            base = (seq_len / (2 * math.pi)) ** (self.dim / (dim_c - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin
    ################## end ## remove input_ids #################


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # print(f'q.shape={q.shape}, k.shape={k.shape}, cos.shape={cos.shape}, sin.shape={sin.shape}, ', flush=True)

    ################# begin # pe for query #################
    q_len = q.shape[2]    
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    ################# begin # pe for query #################

    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



class LightCacheConfig():

    def __init__(
            self, 
            num_key_value_heads, 
            num_attention_heads,
            num_layers,
            
            global_size=32, 
            mid_size=1, 
            span_size=32, 
            local_size=4096,
            chunk_size=512, 
            recall_clip=-1, 

            recall_type='qk',  # ['qk', 'qk_pe', 'k', 'v', 'qkv', 'qkv_pe', 'qkv2', 'qkv2_pe', ]
            pe_original=True, 
            first_prefill=8192, 
            q_cache_len=0, 
            
            rope_scaling=None, 
            score_record=False, 
            recall_option='whole',  # ['default', 'generate_only', 'prefill_only', 'full_attn', ]
            unique_option='group_unique',  # ['head_unique', 'group_unique', 'none', ]

            use_fill=False,
            use_full_attn=False,

            **kwargs
        ):
        self.num_key_value_heads = num_key_value_heads
        self.num_attention_heads = num_attention_heads
        self.kv_group_num = num_attention_heads // num_key_value_heads
        self.num_layers = num_layers

        from functools import partial
        process_to_list = partial(self._process_to_list, num_repeat=num_layers)

        self.global_size = process_to_list(global_size)
        
        self.mid_size = process_to_list(mid_size)
        self.span_size = process_to_list(span_size)
        self.local_size = process_to_list(local_size)
        self.chunk_size = chunk_size
        self.recall_clip = process_to_list(recall_clip)

        self.recall_type = recall_type
        self.pe_original = process_to_list(pe_original)
        self.first_prefill = process_to_list(first_prefill)
        self.q_cache_len = process_to_list(q_cache_len)

        self.rope_scaling = rope_scaling
        self.score_record = score_record
        self.recall_option = process_to_list(recall_option)
        self.unique_option = process_to_list(unique_option)

        self.use_fill = process_to_list(use_fill)
        self.use_full_attn = process_to_list(use_full_attn)
        
        self.additional_params = kwargs

    def _process_to_list(self, x, num_repeat)-> list:
        if not isinstance(x, list):
            return [x] * num_repeat
        assert len(x) == num_repeat
        return x

    def to_dict(self):
        return {**self.__dict__, **self.additional_params}
    
    def __str__(self):
        text = f"AttnCacheConfig:"
        for k, v in self.to_dict().items():
            text += f"\n\t{k}: {v}"
        return text


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@dataclass
class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        raise NotImplementedError("Make sure to implement `get_max_length` in a subclass.")

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    @property
    def seen_tokens(self):
        logger.warning_once(
            "The `seen_tokens` attribute is deprecated and will be removed in v4.41. Use the `cache_position` "
            "model input instead."
        )
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None


@dataclass
class CacheConfig:
    """
    Base class for cache configs
    """

    cache_implementation: None

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Constructs a CacheConfig instance from a dictionary of parameters.
        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration parameters.
            **kwargs: Additional keyword arguments to override dictionary values.
        Returns:
            CacheConfig: Instance of CacheConfig constructed from the dictionary.
        """
        config = cls(**config_dict)
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        return config

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.to_json_file
    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default
                `QuantizationConfig()` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.to_dict
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return copy.deepcopy(self.__dict__)

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.__iter__
    def __iter__(self):
        """allows `dict(obj)` for situations where obj may be a dict or QuantizationConfigMixin"""
        for attr, value in copy.deepcopy(self.__dict__).items():
            yield attr, value

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.__repr__
    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self):
        """
        Serializes this instance to a JSON formatted string.
        Returns:
            str: JSON formatted string representing the configuration instance.
        """
        return json.dumps(self.__dict__, indent=2) + "\n"

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.update
    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs


@dataclass
class QuantizedCacheConfig(CacheConfig):
    """
    Configuration class for quantized cache settings.

    Attributes:
        backend (`str`, *optional*, defaults to `"quanto"`):
            Backend to use when performing quantization, Can be one of [`quanto`, `HQQ`]
        nbits (`Optional[int]`, *optional*, defaults to 4):
            Number of bits, can be 2 or 4 for the `quanto` backend and one of [1, 2, 3, 4, 8] for the `HQQ` backend. Defaults to 2.
        axis_key (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the key tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        axis_value (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the value tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        q_group_size (`Optional[int]`, *optional*, defaults to 64):
            Size of the quantization group, should be a divisor of the model's hidden dimension.
            Defaults to 64.
        residual_length (`Optional[int]`, *optional*, defaults to 128):
            Length of the residual cache which will always be stored in original presicion.
            Defaults to 128.
        compute_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The defualt dtype used for computations in the model. Keys and Values will be cast to this dtype after dequantization.
        device (`str`, *optional*, defaults to `"cpu"`):
            Device on which to peform computations, should be same as the model's device.
    """

    def __init__(
        self,
        backend: str = "quanto",
        nbits: Optional[int] = 4,
        axis_key: Optional[int] = 0,
        axis_value: Optional[int] = 0,
        q_group_size: Optional[int] = 64,
        residual_length: Optional[int] = 128,
        compute_dtype: Optional[torch.dtype] = torch.float16,
        device: Optional[str] = "cpu",
    ):
        self.backend = backend
        self.nbits = nbits
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size
        self.residual_length = residual_length
        self.compute_dtype = compute_dtype
        self.device = device

    def validate(self):
        """Validates if the arguments passed are correct"""

        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        # Check that the values are reasonable in general (nbits, axis)
        # Later in QuantizedCache init we check if they are supported for that particular backend
        if self.nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="nbits",
                    correct_value="2 or 4 or 8",
                    found_value=self.nbits,
                ),
            )
        if self.q_group_size <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="q_group_size",
                    correct_value="a positive integer",
                    found_value=self.q_group_size,
                ),
            )
        if self.residual_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="residual_length",
                    correct_value="a positive integer",
                    found_value=self.residual_length,
                ),
            )

        if self.axis_key not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_key",
                    correct_value="`1` or `0`, `-1`",
                    found_value=self.axis_key,
                ),
            )

        if self.axis_value not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_value",
                    correct_value="`1` or `0` or `-1`",
                    found_value=self.axis_value,
                ),
            )


class CacheList(List):
    
    def __init__(self):
        return super().__init__()
    
    @property
    def shape(self):
        batch_size, head_num, global_len, head_dim = tuple(self[0].shape)
        return (batch_size, head_num, global_len + self[1].shape[-2] + self[2].shape[-2], head_dim)


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self, long_cache_config: LightCacheConfig) -> None:
        self.cache_config = long_cache_config
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.query_cache: List[torch.Tensor] = []
        self.ids_cache: List[torch.Tensor] = []

        # print(f"{self.cache_config.recall_type=}")        

        if 'pe' in self.cache_config.recall_type:
            self.rotary_emb = LlamaScalingNTKScalingRotaryEmbedding(
                    128,  # self.cache_config.additional_params['dim'],  # 
                    max_position_embeddings=self.cache_config.additional_params.get('max_position_embeddings', 8192),  # 8192,
                    base=self.cache_config.additional_params.get('rope_theta', 500000.0),  # 500000,
                )
        else:
            self.rotary_emb = None

        if self.cache_config.score_record:
            self.score_w_pe: List[torch.Tensor] = []
            self.score_wo_pe: List[torch.Tensor] = []
        self.hit_in_topk: List[torch.Tensor] = []

        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self._generate_tokens = 0

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def clean_cache(self):
        del self.key_cache
        self.key_cache = []
        del self.value_cache
        self.value_cache = []
        del self.query_cache
        self.query_cache = []
        if self.cache_config.score_record:
            del self.score_w_pe
            self.score_w_pe = []
            del self.score_wo_pe
            self.score_wo_pe = []
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self._generate_tokens = 0
        torch.cuda.empty_cache()
        print(f"cleared cache.")
        return self
        
    def check_recall(self, is_generate, layer_idx):
        if self._seen_tokens <= self.cache_config.global_size[layer_idx] + self.cache_config.local_size[layer_idx] \
            or (self.cache_config.recall_option[layer_idx] == 'generate_only' and not is_generate) \
            or (self.cache_config.recall_option[layer_idx] == 'prefill_only' and is_generate) \
            or self.cache_config.recall_option[layer_idx] == 'full_attn':
            return False
        else:
            return True
        
    # @torch.compile
    def recall_operation_for_full(self, indices, local_size, layer_idx):
        offsets = torch.arange(-self.cache_config.span_size[layer_idx] // 2, (self.cache_config.span_size[layer_idx] + 1) // 2, device=indices.device)
        fetch_seq_ids = (indices[:, None] + offsets[None, :]).view(-1).contiguous()
        fetch_seq_ids = fetch_seq_ids.clamp(0, self._seen_tokens - 1 - self.cache_config.global_size[layer_idx] - local_size)
        fetch_seq_ids = torch.unique(fetch_seq_ids) + self.cache_config.global_size[layer_idx]
        fetch_seq_ids, _ = torch.sort(fetch_seq_ids, dim=-1)
        return fetch_seq_ids
                    
    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            if self.cache_config.q_cache_len[layer_idx] > 1:
                self.query_cache.append(query_states[..., -self.cache_config.q_cache_len[layer_idx]:, :])
            return self.key_cache[layer_idx], self.value_cache[layer_idx], None, None
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            
        if not self.check_recall(is_generate=(query_states.shape[-2] == 1), layer_idx=layer_idx):
            if self.cache_config.q_cache_len[layer_idx] > 1:
                self.query_cache[layer_idx] = query_states[..., -self.cache_config.q_cache_len[layer_idx]:, :]
            return self.key_cache[layer_idx], self.value_cache[layer_idx], None, None
        elif self.cache_config.mid_size[layer_idx] == 0:
            key_cache = torch.cat([self.key_cache[layer_idx][..., :self.cache_config.global_size[layer_idx], :], 
                                                   self.key_cache[layer_idx][..., -self.cache_config.local_size[layer_idx]:, :], ], dim=-2)
            value_cache = torch.cat([self.value_cache[layer_idx][..., :self.cache_config.global_size[layer_idx], :], 
                                                     self.value_cache[layer_idx][..., -self.cache_config.local_size[layer_idx]:, :]], dim=-2)
            return key_cache, value_cache, None, None
        else:
            batch, num_attention_heads, qlen, head_dim = query_states.shape
            num_key_value_heads, kv_group_num = key_states.shape[1], num_attention_heads // key_states.shape[1]
            
            if qlen == 1 and layer_idx == 0:
                self._generate_tokens += 1

            # [global, mid, local]
            # 下面操作缩短local size让mid size为128的整数，方便算子操作
            global_size = self.cache_config.global_size[layer_idx]
            local_size = self.cache_config.local_size[layer_idx]
            mod_len = (self._seen_tokens - global_size - local_size) % 128
            local_size = local_size - (128 - mod_len)
            mid_size = max(self._seen_tokens - global_size - local_size, 0)

            key_global = self.key_cache[layer_idx][..., :global_size, :]
            value_global = self.value_cache[layer_idx][..., :global_size, :]
            key_local = self.key_cache[layer_idx][..., -local_size:, :]
            value_local = self.value_cache[layer_idx][..., -local_size:, :]
            if self.cache_config.q_cache_len[layer_idx] > 1:
                self.query_cache[layer_idx] = query_states[..., -self.cache_config.q_cache_len[layer_idx]:, :] \
                    if qlen != 1 else torch.cat([self.query_cache[layer_idx][..., 1:, :], query_states], dim=-2)
                query_states = query_states if qlen != 1 else self.query_cache[layer_idx]
            
            # recall_type in ['qk', 'qk_pe', 'k', 'v', 'qkv', 'qkv_pe', 'qkv2', 'qkv2_pe', ] 

            recall_type = self.cache_config.recall_type #[layer_idx]
            if recall_type in ['qk', 'qk_pe']:
                recall_q = query_states
                recall_k = self.key_cache[layer_idx]
            elif recall_type in ['qkv', 'qkv_pe']:
                recall_q = query_states
                recall_k = self.key_cache[layer_idx] * torch.norm(self.value_cache[layer_idx], p=1, dim=-1, keepdim=True)
            elif recall_type in ['qkv2', 'qkv2_pe']:
                recall_q = query_states
                recall_k = self.key_cache[layer_idx] * torch.norm(self.value_cache[layer_idx], p=2, dim=-1, keepdim=True)
            elif recall_type == 'k':
                recall_q = key_states
                recall_k = self.key_cache[layer_idx]
            elif recall_type == 'v':
                recall_q = value_states
                recall_k = self.value_cache[layer_idx]
            else:
                print('not supported yet !', flush=True)
                import sys; sys.exit()

            
            if 'pe' in recall_type:
                cos, sin = self.rotary_emb(self.value_cache[layer_idx])  # , position_ids

                recall_q, recall_k = apply_rotary_pos_emb(recall_q, recall_k, cos.to(query_states.device), sin.to(query_states.device))

            if qlen != 1 and self.cache_config.mid_size[layer_idx] in [1, 4]:
                indices = einsum_topk_func(recall_q, recall_k[..., global_size:-local_size, :], 
                                           min(self.cache_config.mid_size[layer_idx], mid_size))  # b, n, sq, k; 
            else:
                if self.cache_config.recall_type not in ['k', 'v']:
                    recall_q = recall_q.reshape(batch, num_key_value_heads, kv_group_num, qlen, head_dim)
                    recall_q = recall_q.reshape(kv_group_num * batch, num_key_value_heads, qlen, head_dim)

                scores = torch.einsum("bnie,bnje->bnji", recall_q, recall_k[..., global_size:-local_size, :])
                _, indices = torch.topk(scores, min(self.cache_config.mid_size[layer_idx], mid_size), dim=-2)  # 
            
            if self.cache_config.recall_clip[layer_idx] < 0:
                indices = torch.unique(indices)
            else:
                indices, counts = torch.unique(indices, return_counts=True)
                if indices.shape[-1] > self.cache_config.recall_clip[layer_idx]:
                    _, indices_ids = torch.topk(counts, k=self.cache_config.recall_clip[layer_idx])
                    indices = indices[indices_ids]

            fetch_seq_ids = self.recall_operation_for_full(indices, local_size, layer_idx)

            idx_reserved_key = fetch_seq_ids[None, None, :, None].expand(-1, num_key_value_heads, -1, head_dim)
            idx_reserved_value = fetch_seq_ids[None, None, :, None].expand(-1, num_key_value_heads, -1, head_dim)

            key_select = torch.gather(self.key_cache[layer_idx], dim=-2, index=idx_reserved_key)
            value_select = torch.gather(self.value_cache[layer_idx], dim=-2, index=idx_reserved_value)

            global_range = torch.arange(global_size).to(fetch_seq_ids.device)
            local_range = torch.arange(self._seen_tokens - local_size, self._seen_tokens).to(fetch_seq_ids.device)
            position_ids_real = torch.cat([global_range, fetch_seq_ids, local_range])

            key_ret = torch.cat([key_global, key_select, key_local], dim=-2).contiguous()
            value_ret = torch.cat([value_global, value_select, value_local], dim=-2).contiguous()
            
            ret_fill_dict = None
            if hasattr(self.cache_config, 'use_fill') and self.cache_config.use_fill[layer_idx] == True and self._seen_tokens > position_ids_real.shape[0]:
                # print('use cache fill', flush=True)

                total_range = torch.arange(self._seen_tokens).to(fetch_seq_ids.device)
                mask = torch.isin(total_range, position_ids_real, invert=True)
                fill_position = total_range[mask]
                
                ret_fill_dict = {
                    'fill_position': fill_position,
                }

                key_ret = self.key_cache[layer_idx]
                value_ret = self.value_cache[layer_idx]
                return key_ret, \
                    value_ret, \
                    None, None, ret_fill_dict

            #     key_select_remain = self.key_cache[layer_idx][..., fill_position, :]
                
            #     # value_select_remain = self.value_cache[layer_idx][..., fill_position, :]
                
            #     # if query_states.shape[-2] == 1:
            #     #     save_dict = {
            #     #         'select_position': position_ids_real,
            #     #         'fill_position': fill_position,
            #     #         'k': self.key_cache[layer_idx],
            #     #         'v': self.value_cache[layer_idx],
            #     #         'q': query_states,
            #     #     }
            #     #     save_root = './watch_for_reAttn'
            #     #     save_data_dir = os.path.join(save_root, 'data')
            #     #     os.makedirs(save_data_dir, exist_ok=True)
            #     #     save_dict_path = os.path.join(save_data_dir, f'{layer_idx:02d}.pt')
            #     #     torch.save(save_dict, save_dict_path)
            #     #     print(f'{query_states.shape=}')
            #     #     print(f'save data to {save_dict_path}', flush=True)
            #     #     if layer_idx == 31:
            #     #         exit()
                
                
            #     key_states_mean = torch.mean(key_select_remain, dim=-2, keepdim=True)
            #     # value_states_mean = torch.mean(value_select_remain, dim=-2, keepdim=True)
                
            #     key_ret = self.key_cache[layer_idx]
            #     key_ret[..., fill_position, :] = key_states_mean * 0.8
            #     value_ret = self.value_cache[layer_idx]
            #     # value_ret[..., fill_position, :] = value_states_mean
            #     return key_ret, \
            #         value_ret, \
            #         None, None

            if self.cache_config.pe_original[layer_idx]:
                position_ids_for_pe = position_ids_real + 0
                position_ids_for_pe = position_ids_for_pe[None, :]
            else:
                position_ids_for_pe = None
                
            
                
            if hasattr(self.cache_config, 'use_full_attn') and self.cache_config.use_full_attn[layer_idx]:
                key_ret = self.key_cache[layer_idx]
                value_ret = self.value_cache[layer_idx]
                return key_ret, \
                    value_ret, \
                    None, None

            return key_ret, \
                value_ret, \
                position_ids_real[None, :], position_ids_for_pe, ret_fill_dict

    def update_score_w_pe(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor, 
        position_ids: torch.Tensor, 
        layer_idx: int, 
    ):

        if position_ids is not None:
            if len(self.hit_in_topk) <= layer_idx:
                self.hit_in_topk.append([position_ids])
            else:
                self.hit_in_topk[layer_idx] = self.hit_in_topk[layer_idx] + [position_ids]

        b, nq, sq, d = query_states.shape
        b, nk, sk, d = key_states.shape
        sparse_score = torch.zeros((b, self._seen_tokens)).to(query_states.device)

        dense_score = torch.einsum('bnid,bnjd->bnij', query_states, repeat_kv(key_states, query_states.shape[1] // key_states.shape[1]))
        dense_score = torch.nn.functional.softmax(dense_score / math.sqrt(query_states.shape[-1]), dim=-1).sum(-2).sum(1)

        if position_ids is not None:
            sparse_score[:, position_ids] = dense_score.to(sparse_score.dtype)
        else:
            sparse_score = dense_score.to(sparse_score.dtype)

        if len(self.score_w_pe) <= layer_idx:
            self.score_w_pe.append([sparse_score])
        else:
            self.score_w_pe[layer_idx] = self.score_w_pe[layer_idx] + [sparse_score]
        # # batch, num_key_value_heads * n_rep, slen, head_dim
        # # new_score = torch.einsum('bnid,bnjd->bnij', query_states, repeat_kv(key_states, query_states.shape[1] // key_states.shape[1]))
        # # new_score = torch.nn.functional.softmax(new_score / math.sqrt(query_states.shape[-1]), dim=-1).sum(-2)
        # if len(self.score_w_pe) <= layer_idx:
        #     b, nq, sq, d = query_states.shape
        #     b, nk, sk, d = key_states.shape
        #     new_score = torch.zeros((b, nq, sk)).to(query_states)
        #     self.score_w_pe.append(new_score)
        # else:
        #     seq_len = self.score_w_pe[layer_idx].shape[-1]
        #     new_score = torch.einsum('bnid,bnjd->bnij', query_states, repeat_kv(key_states[..., :seq_len, :], query_states.shape[1] // key_states.shape[1]))
        #     new_score = torch.nn.functional.softmax(new_score / math.sqrt(query_states.shape[-1]), dim=-1).sum(-2)
        #     self.score_w_pe[layer_idx] = self.score_w_pe[layer_idx] + new_score
        
    def update_score_wo_pe(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_ids: torch.Tensor, 
        layer_idx: int, 
    ):
        b, nq, sq, d = query_states.shape
        b, nk, sk, d = key_states.shape
        sparse_score = torch.zeros((b, self._seen_tokens)).to(query_states.device)

        dense_score = torch.einsum('bnid,bnjd->bnij', query_states, repeat_kv(key_states, query_states.shape[1] // key_states.shape[1]))
        dense_score = torch.nn.functional.softmax(dense_score / math.sqrt(query_states.shape[-1]), dim=-1).sum(-2).sum(1)

        if position_ids is not None:
            sparse_score[:, position_ids] = dense_score.to(sparse_score.dtype)
        else:
            sparse_score = dense_score.to(sparse_score.dtype)

        if len(self.score_wo_pe) <= layer_idx:
            self.score_wo_pe.append([sparse_score])
        else:
            self.score_wo_pe[layer_idx] = self.score_wo_pe[layer_idx] + [sparse_score]
        
    def save_score(
        self, 
        save_root: str,
        # box_pos: list = None, 
    ):
        # num_col, nol_row = 4, math.ceil(len(self.score_wo_pe) / 4)
        if self.cache_config.score_record:
            save_dict = {'score_wo_pe': self.score_wo_pe, 'score_w_pe': self.score_w_pe, }
            torch.save(save_dict, save_root)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        return self._seen_tokens  # self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, long_cache_config: LightCacheConfig, 
                          past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, ) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""
        cache = cls(long_cache_config=long_cache_config)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def crop(self, maximum_length: int):
        """Crop the past key values up to a new `maximum_length` in terms of tokens. `maximum_length` can also be
        negative to remove `maximum_length` tokens. This is used in assisted decoding and contrastive search."""

        # In case it is negative
        if maximum_length < 0:
            maximum_length = self.get_seq_length() - abs(maximum_length)

        if self.get_seq_length() <= maximum_length:
            return

        self._seen_tokens = maximum_length
        for idx in range(len(self.key_cache)):
            self.key_cache[idx] = self.key_cache[idx][..., :maximum_length, :]
            self.value_cache[idx] = self.value_cache[idx][..., :maximum_length, :]

    def batch_split(self, full_batch_size: int, split_size: int) -> List["DynamicCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = DynamicCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["DynamicCache"]) -> "DynamicCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls()
        for idx in range(len(splits[0])):
            layer_keys = torch.cat([current.key_cache[idx] for current in splits], dim=0)
            layer_values = torch.cat([current.value_cache[idx] for current in splits], dim=0)
            cache.update(layer_keys, layer_values, idx)
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]


class QuantizedCache(DynamicCache):
    """
    A quantizer cache similar to what is described in the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://arxiv.org/abs/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for Key and Value cache by applying quantization.

    The cache has two types of storage, one for original precision and one for the quantized cache. A `residual length` is set as a maximum capacity for the
    original precision cache. When the length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache. The
    quantization is done per-channel with a set `q_group_size` for both Keys and Values, in contrast to what was described in the paper.

    It stores Keys and Values a list of quantized tensors (tuples in case we need to store metadata), one for each layer. Additionally, it stores the Key and
    Value in original precision states as a list of tensors, one for each layer. The size of each tensor
    is `[batch_size, num_heads, seq_len - residual_length, head_dim]`
    """

    def __init__(self, cache_config: QuantizedCacheConfig) -> None:
        self._quantized_key_cache: List[torch.Tensor] = []
        self._quantized_value_cache: List[torch.Tensor] = []

        self.nbits = cache_config.nbits
        self.residual_length = cache_config.residual_length
        self.q_group_size = cache_config.q_group_size
        self.axis_key = cache_config.axis_key
        self.axis_value = cache_config.axis_value
        self.compute_dtype = cache_config.compute_dtype
        self.device = cache_config.device

        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            self._quantized_key_cache.append(self._quantize(key_states.contiguous(), axis=self.axis_key))
            self._quantized_value_cache.append(self._quantize(value_states.contiguous(), axis=self.axis_value))
            self.key_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
            self.value_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
            keys_to_return, values_to_return = key_states, value_states
        else:
            dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
            dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
            keys_to_return = [dequant_key, self.key_cache[layer_idx], key_states]
            values_to_return = [dequant_value, self.value_cache[layer_idx], value_states]

            keys_to_return = torch.cat(keys_to_return, dim=-2)
            values_to_return = torch.cat(values_to_return, dim=-2)
            if (
                self.key_cache[layer_idx].dim() == 4
                and self.key_cache[layer_idx].shape[-2] + 1 >= self.residual_length
            ):
                self._quantized_key_cache[layer_idx] = self._quantize(keys_to_return.contiguous(), axis=self.axis_key)
                self._quantized_value_cache[layer_idx] = self._quantize(
                    values_to_return.contiguous(), axis=self.axis_value
                )
                self.key_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
                self.value_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and rely on `_seen_tokens` which is
        # updated every "layer_idx" == 0, this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def _quantize(self, tensor, axis):
        """Quantizes a key/value using a defined quantization method."""
        raise NotImplementedError("Make sure to implement `_quantize` in a subclass.")

    def _dequantize(self, q_tensor):
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""
        raise NotImplementedError("Make sure to implement `_dequantize` in a subclass.")


class QuantoQuantizedCache(QuantizedCache):
    """
    Quantized Cache class that uses `quanto` as a backend to perform quantization. Current implementation supports `int2` and `int4` dtypes only.

    Parameters:
        cache_config (`QuantizedCacheConfig`,):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.
    """

    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
        quanto_version = version.parse(importlib.metadata.version("quanto"))
        if quanto_version < version.parse("0.2.0"):
            raise ImportError(
                f"You need quanto package version to be greater or equal than 0.2.0 to use `QuantoQuantizedCache`. Detected version {quanto_version}. "
                f"Please upgrade quanto with `pip install -U quanto`"
            )

        if self.nbits not in [2, 4]:
            raise ValueError(f"`nbits` for `quanto` backend has to be one of [`2`, `4`] but got {self.nbits}")

        if self.axis_key not in [0, -1]:
            raise ValueError(f"`axis_key` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_key}")

        if self.axis_value not in [0, -1]:
            raise ValueError(
                f"`axis_value` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_value}"
            )

        self.qtype = qint4 if self.nbits == 4 else qint2
        self.optimizer = MaxOptimizer()  # hardcode as it's the only one for per-channel quantization

    def _quantize(self, tensor, axis):
        scale, zeropoint = self.optimizer(tensor, self.qtype.bits, axis, self.q_group_size)
        qtensor = AffineQuantizer.apply(tensor, self.qtype, axis, self.q_group_size, scale, zeropoint)
        return qtensor

    def _dequantize(self, qtensor):
        return qtensor.dequantize()


class HQQQuantizedCache(QuantizedCache):
    """
    Quantized Cache class that uses `HQQ` as a backend to perform quantization. Current implementation supports `int2`, `int4`, `int8` dtypes.

    Parameters:
        cache_config (`QuantizedCacheConfig`,):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.
    """

    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
        if self.nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                f"`nbits` for `HQQ` backend has to be one of [`1`, `2`, `3`, `4`, `8`] but got {self.nbits}"
            )

        if self.axis_key not in [0, 1]:
            raise ValueError(f"`axis_key` for `HQQ` backend has to be one of [`0`, `1`] but got {self.axis_key}")

        if self.axis_value not in [0, 1]:
            raise ValueError(f"`axis_value` for `HQQ` backend has to be one of [`0`, `1`] but got {self.axis_value}")

        self.quantizer = HQQQuantizer

    def _quantize(self, tensor, axis):
        qtensor, meta = self.quantizer.quantize(
            tensor,
            axis=axis,
            device=self.device,
            compute_dtype=self.compute_dtype,
            nbits=self.nbits,
            group_size=self.q_group_size,
        )
        meta["compute_dtype"] = self.compute_dtype
        self.quantizer.cuda(qtensor, meta=meta, device=self.device)  # Move to device and cast to dtype
        return qtensor, meta

    def _dequantize(self, qtensor):
        quant_tensor, meta = qtensor
        tensor = self.quantizer.dequantize(quant_tensor, meta)
        return tensor


class SinkCache(Cache):
    """
    A cache that as described in the [Attention Sinks paper](https://arxiv.org/abs/2309.17453). It allows the model to
    generate beyond the length of its context window, without losing fluency in the conversation. As it discards past
    tokens, the model will lose the ability to generate tokens that depend on the context that was discarded.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Parameters:
        window_length (`int`):
            The length of the context window.
        num_sink_tokens (`int`):
            The number of sink tokens. See the original paper for more information.
    """

    def __init__(self, window_length: int, num_sink_tokens: int) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self.cos_sin_rerotation_cache = {}
        self._cos_cache = None
        self._sin_cache = None
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_key_rotary_pos_emb(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        rotated_key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return rotated_key_states

    def _get_rerotation_cos_sin(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape[-2] not in self.cos_sin_rerotation_cache:
            # Upcast to float32 temporarily for better accuracy
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)

            # Compute the cos and sin required for back- and forward-rotating to one position earlier in the sequence
            original_cos = cos[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_cos = cos[self.num_sink_tokens : -key_states.shape[-2]]
            original_sin = sin[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_sin = sin[self.num_sink_tokens : -key_states.shape[-2]]
            rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
            rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin

            self.cos_sin_rerotation_cache[key_states.shape[-2]] = (
                rerotation_cos.to(key_states.dtype).unsqueeze(0),
                rerotation_sin.to(key_states.dtype).unsqueeze(0),
            )
        return self.cos_sin_rerotation_cache[key_states.shape[-2]]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.window_length

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The following arguments can be used in `SinkCache`: `sin`,
                `cos` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
                rotation as the tokens are shifted.

        Return:
            A tuple containing the updated key and value states.
        """
        # Optional kwargs for `SinkCache` -- needed on models using RoPE. `partial_rotation_size` is used on models
        # with partially rotated position embeddings, like Phi or Persimmon.
        sin = cache_kwargs.get("sin")
        cos = cache_kwargs.get("cos")
        partial_rotation_size = cache_kwargs.get("partial_rotation_size")
        using_rope = cos is not None and sin is not None

        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the sin/cos cache, which holds sin/cos values for all possible positions
        if using_rope and layer_idx == 0:
            # BC: some models still pass `sin`/`cos` with 2 dims. In those models, they are the full sin/cos. Remove
            # after all RoPE models have a llama-like cache utilization.
            if cos.dim() == 2:
                self._cos_cache = cos
                self._sin_cache = sin
            else:
                if self._cos_cache is None:
                    self._cos_cache = cos[0, ...]
                    self._sin_cache = sin[0, ...]
                elif self._cos_cache.shape[0] < self.window_length:
                    self._cos_cache = torch.cat([self._cos_cache, cos[0, ...]], dim=0)
                    self._sin_cache = torch.cat([self._sin_cache, sin[0, ...]], dim=0)

        # [bsz, num_heads, seq_len, head_dim]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        elif key_states.shape[-2] + self.get_seq_length(layer_idx) < self.window_length:
            # Growing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        else:
            # Shifting cache
            keys_to_keep = self.key_cache[layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + key_states.shape[-2] :
            ]

            # On RoPE models, we need to recompute the Key rotation as the tokens are shifted
            if using_rope:
                rerotation_cos, rerotation_sin = self._get_rerotation_cos_sin(
                    key_states, self._cos_cache[: self.window_length], self._sin_cache[: self.window_length]
                )
                if partial_rotation_size is not None:
                    keys_to_keep, keys_pass = (
                        keys_to_keep[..., :partial_rotation_size],
                        keys_to_keep[..., partial_rotation_size:],
                    )
                keys_to_keep = self._apply_key_rotary_pos_emb(keys_to_keep, rerotation_cos, rerotation_sin)
                if partial_rotation_size is not None:
                    keys_to_keep = torch.cat((keys_to_keep, keys_pass), dim=-1)

            # Concatenate sink tokens, shifted & rotated tokens (if needed), and new tokens
            sink_keys = self.key_cache[layer_idx][:, :, : self.num_sink_tokens]
            self.key_cache[layer_idx] = torch.cat([sink_keys, keys_to_keep, key_states], dim=-2)

            sink_values = self.value_cache[layer_idx][:, :, : self.num_sink_tokens]
            values_to_keep = self.value_cache[layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + value_states.shape[-2] :
            ]
            self.value_cache[layer_idx] = torch.cat([sink_values, values_to_keep, value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)`.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
    """

    def __init__(self, config: PretrainedConfig, max_batch_size: int, max_cache_len: int, device, dtype=None) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        cache_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        for _ in range(config.num_hidden_layers):
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()


class SlidingWindowCache(StaticCache):
    """
    Sliding Window Cache class to be used with `torch.compile` for models like Mistral that support sliding window attention.
    Every time when we try to update the cache, we compute the `indices` based on `cache_position >= self.config.sliding_window - 1`,
    if true(which means the cache can not hold all the old key value states and new states together because of the sliding window constraint),
    we need to do a cycle shift based on `indices` to replace the oldest states by the new key value states passed in.

    The `to_shift` is only true once we are above sliding_window. Thus with `sliding_window==64`:

    indices = (slicing + to_shift[-1].int()-1) % self.config.sliding_window
    tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63,  0])

    We overwrite the cache using these, then we always write at cache_position (clamped to `sliding_window`)

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
    """

    def __init__(self, config: PretrainedConfig, max_batch_size: int, max_cache_len: int, device, dtype=None) -> None:
        if not hasattr(config, "sliding_window") or config.sliding_window is None:
            raise ValueError(
                "Setting `cache_implementation` to 'sliding_window' requires the model config supporting "
                "sliding window attention, please check if there is a `sliding_window` field in the model "
                "config and it's not set to None."
            )
        max_cache_len = min(config.sliding_window, max_cache_len)
        super().__init__(
            config=config, max_batch_size=max_batch_size, max_cache_len=max_cache_len, device=device, dtype=dtype
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor]:
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        # assume this only happens in prefill phase when prompt length > sliding_window_size (= max_cache_len)
        if cache_position.shape[0] > self.max_cache_len:
            k_out = key_states[:, :, -self.max_cache_len :, :]
            v_out = value_states[:, :, -self.max_cache_len :, :]
            # Assumption: caches are all zeros at this point, `+=` is equivalent to `=` but compile-friendly
            self.key_cache[layer_idx] += k_out
            self.value_cache[layer_idx] += v_out
            # we should return the whole states instead of k_out, v_out to take the whole prompt
            # into consideration when building kv cache instead of just throwing away tokens outside of the window
            return key_states, value_states

        slicing = torch.ones(self.max_cache_len, dtype=torch.long, device=value_states.device).cumsum(0)
        cache_position = cache_position.clamp(0, self.max_cache_len - 1)
        to_shift = cache_position >= self.max_cache_len - 1
        indices = (slicing + to_shift[-1].int() - 1) % self.max_cache_len

        k_out = k_out[:, :, indices]
        v_out = v_out[:, :, indices]

        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states

        # `_.zero()` followed by `+=` is equivalent `=`, but compile-friendly (without graph breaks due to assignment)
        self.key_cache[layer_idx].zero_()
        self.value_cache[layer_idx].zero_()

        self.key_cache[layer_idx] += k_out
        self.value_cache[layer_idx] += v_out

        return k_out, v_out

    def get_max_length(self) -> Optional[int]:
        # in theory there is no limit because the sliding window size is fixed
        # no matter how long the sentence is
        return None