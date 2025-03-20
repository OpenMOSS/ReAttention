import os
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import transformers

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
import torch.nn.functional as F

from transformers import AutoConfig
from transformers import AutoTokenizer
import datetime
import os

os.environ["VLLM_USE_MODELSCOPE"] = "False"

PromptType = Union[PromptList, str]

from ..huggingface import HuggingFaceCausalLM, BaseModel

from vllm import LLM, SamplingParams


@MODELS.register_module()
class VLLMCausalLM(HuggingFaceCausalLM):
    def __init__(self,
                 path: str,
                 hf_cache_dir: Optional[str] = None,
                 max_seq_len: int = 2048,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 tokenizer_only: bool = False,
                 model_kwargs: dict = dict(device_map='auto'),
                 generation_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False,
                 pad_token_id: Optional[int] = None,
                 mode: str = 'none',
                 use_fastchat_template: bool = False,
                 end_str: Optional[str] = None,
                 long_cache_config = None,
                 long_bench_cat = -1,
                 prompt_format: str = '{prompt}',
                 attn_implementation: str = 'eager', 
                 quanto_enable: bool = False, 
                 chat_enable: bool = False):
        BaseModel.__init__(self, path=path,
                         max_seq_len=max_seq_len,
                         tokenizer_only=tokenizer_only,
                         meta_template=meta_template)
        if hf_cache_dir is None:
            hf_cache_dir = os.getenv('HF_MODEL_HUB', None)
        self.logger = get_logger()
        self.pad_token_id = pad_token_id
        assert mode in ['none', 'mid']
        self.mode = mode

        self.long_cache_config = long_cache_config
        self.attn_implementation = attn_implementation

        self._load_tokenizer(path=path, 
                             tokenizer_path=tokenizer_path,
                             tokenizer_kwargs=tokenizer_kwargs)
        
        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode
        self.long_bench_cat = long_bench_cat
        if not tokenizer_only:
            self._load_model(path=path,
                             model_kwargs=model_kwargs,
                             peft_path=peft_path)
        self.generation_kwargs = generation_kwargs
        self.use_fastchat_template = use_fastchat_template
        self.end_str = end_str

        self.prompt_format = prompt_format
                
        self.quanto_enable = quanto_enable
        self.chat_enable = chat_enable

    def _load_tokenizer(self, path: Optional[str], tokenizer_path: Optional[str], tokenizer_kwargs: dict):
        
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)  # , local_files_only=True

        super()._load_tokenizer(path=path, tokenizer_path=tokenizer_path, tokenizer_kwargs=tokenizer_kwargs)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.eos_token_id

    def _load_model(self,
                    path: str, 
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        # Create a sampling params object.
        self.sampling_params = SamplingParams(top_k=1)
        # Create an LLM.
        self.llm = LLM(model=path, trust_remote_code=True)
        #    gpu_memory_utilization=0.75, 
        #    max_model_len=(1 + self.long_bench_cat // 1000) * 1000
                       

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]: 
        """Generate results given a list of inputs."""
        outputs_text = []
             
        for text in inputs:
            
            if text.endswith('\n\n') and not self.chat_enable:
                text = text[:-2]
            
            if self.chat_enable:
                input_ids = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], 
                                                            tokenize=True, add_generation_prompt=True, 
                                                            return_tensors="pt")
            else:
                input_ids = self.tokenizer(text, return_tensors="pt").input_ids

            if self.long_bench_cat > 0 and  input_ids.shape[-1] > self.long_bench_cat:
                input_ids = torch.cat([input_ids[:, : self.long_bench_cat // 2], input_ids[:, - self.long_bench_cat // 2:]], dim=-1)
            
            print(f"\n\ninput_ids.shape: {input_ids.shape}\n", flush=True)
            seq_len = input_ids.shape[-1]
            input_ids = input_ids.tolist()

            outputs = self.llm.generate(prompt_token_ids=input_ids, sampling_params=self.sampling_params)
            
            generated_text = outputs[0].outputs[0].text

            outputs_text.append(generated_text)
            torch.cuda.empty_cache()
        
        # print(prof.table())
        # prof.export_chrome_trace('/fs-computility/llm/shared/liuxiaoran/opencompass/opencompass/models/re_attention/resnet_profile2.json')

        return outputs_text