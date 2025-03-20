import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]

# from .huggingface import HuggingFaceCausalLM
from opencompass.models.huggingface import HuggingFaceCausalLM
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
from .selfextend_utils import SelfExtend

@MODELS.register_module()
class SelfExtend_LlamaForCausalLM(HuggingFaceCausalLM):
    def __init__(self, **kwargs):
        # With Llama-2 as the base model, 2~64 are reasonable for group_size; 
        # 512~1536 are feasible for neighbor_window. But larger group_size and smaller neighbor_window 
        # are also good in many cases

        # The general rule of choosing group_size and neighbor_window is: ensure the input sequence 
        # lenght is within the maximum extended window size (For llama-2, it would be 
        # (4096 - neighbor_window) * group_size + neighbor_window ).
        se_dataparam: Dict = kwargs.pop('selfextend_kwargs', None)
        self.selfextend_group_size = 64
        self.selfextend_window_size = 1024
        if se_dataparam is not None:
            self.selfextend_group_size = se_dataparam.get('group_size', 64)
            self.selfextend_window_size = se_dataparam.get('window_size', 1024)
        super().__init__(**kwargs)

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        # from transformers import AutoModelForCausalLM

        self._set_model_kwargs_torch_dtype(model_kwargs)

        loaded_model = LlamaForCausalLM.from_pretrained(path, **model_kwargs)

        SelfExtend.apply(loaded_model, self.selfextend_group_size, self.selfextend_window_size, 
                         enable_flash_attention=True, flash_attention_impl="flash_attn")
        
        self.model = loaded_model
        # self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                    peft_path,
                                                    is_trainable=False)
        
        self.model.eval()
        self.model.generation_config.do_sample = False