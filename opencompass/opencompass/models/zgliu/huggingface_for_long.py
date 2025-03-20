import os
import sys

import argparse

from functools import reduce
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger

from opencompass.models.huggingface import HuggingFaceCausalLM
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig  # PretrainedConfig

# sys.path.append('/cpfs01/shared/public/lvkai/workspace/collie/')

# from collie import CollieConfig
from transformers.generation.utils import GenerationConfig
from transformers import AutoTokenizer


@MODELS.register_module()
class HuggingFaceModelForLong(HuggingFaceCausalLM):

    def __init__(self,
                 *args,
                 **kwargs):
        
        self.long_bench_cat = kwargs.pop('long_bench_cat', -1)
        self.long_cache_config = kwargs.pop('long_cache_config', None)
        self.model_type = kwargs.pop('model_type', 'llama')
        self.prompt_format = kwargs.pop('prompt_format', None)
        self.quanto_enable = kwargs.pop('quanto_enable', False)
        
        super().__init__(*args, **kwargs)

        self.generation_config = GenerationConfig(
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            num_beams=1, do_sample=False, use_cache=True
        )

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        from transformers import AutoModel, AutoModelForCausalLM, AutoConfig

        self._set_model_kwargs_torch_dtype(model_kwargs)
        try:
            config = AutoConfig.from_pretrained(path)
            if self.long_cache_config and self.long_cache_config.get('rope_scaling', None):
                config.rope_scaling = self.long_cache_config['rope_scaling']
            elif model_kwargs.get('rope_scaling', None):
                config.rope_scaling = model_kwargs.pop('rope_scaling')
                
            self.model = AutoModelForCausalLM.from_pretrained(
                path, **model_kwargs, config=config)
        except ValueError:
            self.model = AutoModel.from_pretrained(path, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                   peft_path,
                                                   is_trainable=False)
        self.model.eval()
        self.model.generation_config.do_sample = False

        # A patch for llama when batch_padding = True
        if 'decapoda-research/llama' in path:
            self.model.config.bos_token_id = 1
            self.model.config.eos_token_id = 2
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        

    def _single_generate(self, inputs: List[str], max_out_len: int, **kwargs) -> List[str]:
        """Support for single prompt inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        inputs = [x.strip() for x in inputs]

        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]

        if self.long_bench_cat > 0:
            inputs = [self.prompt_format.format(prompt=prompt) for prompt in inputs]
            input_ids = self.tokenizer(inputs, padding=False, truncation=False)['input_ids']
            input_ids = torch.tensor(input_ids)
            if input_ids.shape[-1] > self.long_bench_cat:
                input_ids = torch.cat([input_ids[:, : self.long_bench_cat // 2], input_ids[:, - self.long_bench_cat // 2:]], dim=-1).to(device=self.model.device)
            else:
                input_ids = input_ids.to(device=self.model.device)
        else:
            input_ids = self.tokenizer(inputs, padding=False, truncation=True, max_length=self.max_seq_len)['input_ids']
            input_ids = torch.tensor(input_ids).to(device=self.model.device)
        
        generation_config = self.generation_config
        generation_config.max_new_tokens = max_out_len
        # self.logger.info('input_ids give')
        outputs = self.model.generate(input_ids=input_ids, generation_config=generation_config)

        if not self.extract_pred_after_decode:
            outputs = outputs[:, input_ids.shape[1]:]
        # self.logger.info('outputs return')
        decodeds = self.tokenizer.batch_decode(outputs.cpu().tolist(), skip_special_tokens=True)

        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]

        return decodeds