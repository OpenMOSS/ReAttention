import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]

# from .huggingface import HuggingFaceCausalLM
from opencompass.models.huggingface import HuggingFaceCausalLM
from .lightcache_utils.light_cache_mosshuawei import MossHuaweiForCausalLM
from .lightcache_utils.cache_utils import LightCacheConfig
from transformers import AutoConfig

@MODELS.register_module()
class LightCache_MossHuaweiForCausalLM(HuggingFaceCausalLM):

    def __init__(self, **kwargs):
        self.long_cache_config_kwargs = kwargs.pop('long_cache_config_kwargs', {})
        super().__init__(**kwargs)

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None,
                    ):
        # from transformers import AutoModelForCausalLM

        self._set_model_kwargs_torch_dtype(model_kwargs)

        config = AutoConfig.from_pretrained(path, trust_remote_code=True)

        long_cache_config = LightCacheConfig(num_key_value_heads=config.num_key_value_heads, 
                                    num_attention_heads=config.num_attention_heads, 
                                    **self.long_cache_config_kwargs)
        
        config.long_cache_config = long_cache_config
        config.pretraining_tp = 1

        self.model = MossHuaweiForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto",
                                              config=config, trust_remote_code=True, 
                                             attn_implementation='flash_attention_2')

        # self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                    peft_path,
                                                    is_trainable=False)
        
        self.model.eval()
        self.model.generation_config.do_sample = False