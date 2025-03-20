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
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from .infllm_utils import patch_hf, GreedySearch, patch_model_center

@MODELS.register_module()
class InfLLM_xrliu(HuggingFaceCausalLM):
    def __init__(self, *args, **kwargs):
        self.infllm_kwargs = kwargs.pop('infllm_kwargs')
        super().__init__(*args, **kwargs)
        self.chunk_size = self.infllm_kwargs['chunk_size']

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        # from transformers import AutoModelForCausalLM

        self._set_model_kwargs_torch_dtype(model_kwargs)

        if self.infllm_kwargs.model_center:
            import bmtrain as bmt
            bmt.init_distributed(seed=233)
            from model_center.model import Llama, LlamaConfig
            model_config = LlamaConfig.from_pretrained(path)
            model_config.dtype = torch.bfloat16
            self.model = Llama(model_config)
            bmt.load(self.model, os.path.join(path, "pytorch_model.pt"), strict=False)
            self.model = patch_model_center(self.model, self.infllm_kwargs.type, **self.infllm_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
            self.model = patch_hf(self.model, self.infllm_kwargs.type, **self.infllm_kwargs)

        # self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        
        self.model.eval()
        self.model.generation_config.do_sample = False
        self.searcher = GreedySearch(self.model, self.tokenizer)
        
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        
        outputs_text = []

        for text in inputs:
            
            if text.endswith('\n\n'):
                text = text[:-2]

            input_ids = self.tokenizer(text, return_tensors="pt").input_ids
                
            if self.max_seq_len > 0:
                if input_ids.shape[-1] > self.max_seq_len:
                    input_ids = torch.cat([input_ids[:, : self.max_seq_len // 2], input_ids[:, - self.max_seq_len // 2:]], dim=-1).to(device=self.model.device)
                else:
                    input_ids = input_ids.to(device=self.model.device)
            else:
                input_ids = input_ids.to(device=self.model.device)

            print(f"\n\ninput_ids.shape: {input_ids.shape}\n", flush=True)
            
            output = self.searcher.generate(
                input_ids=input_ids,
                max_length=max_out_len,
                chunk_size=self.chunk_size,
                extra_end_token_ids=[]
            )
            
            outputs_text.append(output[0])
            torch.cuda.empty_cache()
            
        return outputs_text
