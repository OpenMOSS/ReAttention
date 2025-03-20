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
# from transformers import AutoModelForCausalLM, LlamaForCausalLM
from .rerope_utils import LlamaForCausalLM_4_31_0

@MODELS.register_module()
class Rerope_LlamaForCausalLM(HuggingFaceCausalLM):
    def __init__(self, *args, **kwargs):
        rr_kwargs = kwargs.pop('rerope_kwargs', None)
        if rr_kwargs is not None:
            print("rerope_kwargs is not None")
        super().__init__(*args, **kwargs)

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        # from transformers import AutoModelForCausalLM

        self._set_model_kwargs_torch_dtype(model_kwargs)

        # using LlamaForCausalLM from transformers.4.31.0; no flash attention in this version
        self.model = LlamaForCausalLM_4_31_0.from_pretrained(path, **model_kwargs)

        # self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        
        self.model.eval()
        self.model.generation_config.do_sample = False