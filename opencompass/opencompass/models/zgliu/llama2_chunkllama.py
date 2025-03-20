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
from transformers import LlamaForCausalLM, LlamaConfig
from .chunkllama_utils import replace_with_chunkllama

@MODELS.register_module()
class ChunkLlama_LlamaForCausalLM(HuggingFaceCausalLM):
    def __init__(self, **kwargs):
        cl_kwargs: Dict = kwargs.pop('chunkllama_kwargs', None)
        self.pretraining_length = None
        if cl_kwargs is not None:
            self.pretraining_length = cl_kwargs.get('pretraining_length', None)
        super().__init__(**kwargs)

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        # from transformers import AutoModelForCausalLM

        self._set_model_kwargs_torch_dtype(model_kwargs)
        if self.pretraining_length is None:
            config: LlamaConfig = LlamaConfig.from_pretrained(path)
            self.pretraining_length = config.max_position_embeddings 
            if config.rope_scaling is not None:
                self.pretraining_length = config.rope_scaling.get("original_max_position_embeddings", self.pretraining_length)

        replace_with_chunkllama(self.pretraining_length)
        self.model = LlamaForCausalLM.from_pretrained(path, **model_kwargs)
        
        # self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                    peft_path,
                                                    is_trainable=False)
        
        self.model.eval()
        self.model.generation_config.do_sample = False