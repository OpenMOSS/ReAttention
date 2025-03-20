import os
import sys

import argparse

from functools import reduce
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger

from opencompass.models.base import BaseModel, LMTemplateParser
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig  # PretrainedConfig

# sys.path.append('/cpfs01/shared/public/lvkai/workspace/collie/')

# from collie import CollieConfig
from transformers.generation.utils import GenerationConfig
from transformers import AutoTokenizer


@MODELS.register_module()
class HuggingFaceModel(BaseModel):
    """Model wrapper around HuggingFace CausalLM.

    Args:
        path (str): The name or path to HuggingFace's model.
        hf_cache_dir: Set the cache dir to HF model cache dir. If None, it will
            use the env variable HF_MODEL_HUB. Defaults to None.
        max_seq_len (int): The maximum length of the input sequence. Defaults
            to 2048.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
            Defaults to {}.
        peft_path (str, optional): The name or path to the HuggingFace's PEFT
            model. If None, the original model will not be converted to PEFT.
            Defaults to None.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        model_kwargs (dict): Keyword arguments for the model, used in loader.
            Defaults to dict(device_map='auto').
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        batch_padding (bool): If False, inference with be performed in for-loop
            without batch padding.
    """

    def __init__(self,
                 config_path: str,
                 model_path: str,
                 model_type: str = None,  # ['llama2', 'internlm2', 'chatglm3']
                 max_seq_len: int = 4096, 
                 long_bench_cat: int = -1, 
                 prompt_format: str = '{prompt}', 
                 model_kwargs: dict = dict(), 
                 tokenizer_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False):
        
        super().__init__(path=config_path,
                         max_seq_len=max_seq_len,
                         tokenizer_only=False,
                         meta_template=meta_template)

        self.model_path = model_path
        self.model_type = model_type
        self.batch_padding = batch_padding
        self.long_bench_cat = long_bench_cat
        self.prompt_format = prompt_format
        
        model_config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)  # PretrainedConfig, local_files_only=True
        
        if self.model_type == 'chatglm3':
            model_config.pre_seq_len = None
            model_config.prefix_projection = None
            model_config.quantization_bit = None
        
        self.model_config = model_config
        self._load_tokenizer(tokenizer_path=config_path, tokenizer_kwargs=tokenizer_kwargs)
        self._load_model(model_path=model_path, model_kwargs=model_kwargs, model_config=model_config, model_type=model_type)
        self.logger = get_logger()
        
        self.extract_pred_after_decode = extract_pred_after_decode
        
        self.generation_config = GenerationConfig(
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            num_beams=1, do_sample=False, use_cache=True
        )

    def _load_tokenizer(self, tokenizer_path: Optional[str], tokenizer_kwargs: dict):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)  # , local_files_only=True

        self.eos_token_id = self.tokenizer.eos_token_id

        if self.model_type in ['llama2', 'internlm2', ]: 
            self.tokenizer.bos_token = '<s>'
            self.tokenizer.eos_token = '</s>'
            self.pad_token_id = self.tokenizer.pad_token_id
        else:
            self.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self, model_path: str, model_kwargs, model_config, model_type: str):
        
        # assert model_type == 'other'

        model_config._flash_attn_2_enabled = True
        model_config.attn_implementation = "flash_attention_2"
        # model_config._attn_implementation = "flash_attention_2"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, config=model_config,  # **model_kwargs)
                                                              torch_dtype=torch.float16, device_map="auto",  trust_remote_code=True)  # , local_files_only=True: PretrainedConfig
        except ValueError:
            self.model = AutoModel.from_pretrained(model_path, config=model_config,  # **model_kwargs)
                                                   torch_dtype=torch.float16, device_map="auto",  trust_remote_code=True)
                                                      
        self.model.eval()

    def generate(self, inputs: List[str], max_out_len: int, **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
                
        if self.batch_padding and len(inputs) > 1:
            return self._batch_generate(inputs=inputs, max_out_len=max_out_len, **kwargs)
        else:
            return sum((self._single_generate(
                inputs=[input_], max_out_len=max_out_len, **kwargs)
                for input_ in inputs), [])

    def _batch_generate(self, inputs: List[str], max_out_len: int, **kwargs) -> List[str]:
        """Support for batch prompts inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]

        # step-1: tokenize the input with batch_encode_plus
        tokens = self.tokenizer.batch_encode_plus(inputs, padding=True, truncation=True, 
                                                  max_length=self.max_seq_len - max_out_len)
        tokens = {
            k: torch.tensor(np.array(tokens[k]), device=self.model.device)
            for k in tokens if k in ['input_ids', 'attention_mask']
        }

        # step-2: conduct model forward to generate output
        generation_config = self.generation_config
        generation_config.max_new_tokens = max_out_len
        # self.logger.info('input_ids given')
        outputs = self.model.generate(**tokens, generation_config=generation_config)

        if not self.extract_pred_after_decode:
            outputs = outputs[:, tokens['input_ids'].shape[1]:]
        # self.logger.info('outputs return')
        decodeds = self.tokenizer.batch_decode(outputs.cpu().tolist(), skip_special_tokens=True)

        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]

        return decodeds

    def _single_generate(self, inputs: List[str], max_out_len: int, **kwargs) -> List[str]:
        """Support for single prompt inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
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

    def get_logits(self, inputs: List[str]):

        if self.batch_padding and len(inputs) > 1:
            # batch inference
            tokens = self.tokenizer(inputs, padding=True, truncation=True,
                                    max_length=self.max_seq_len)

            tokens = {
                k: torch.tensor(np.array(tokens[k]), device=self.model.device)
                for k in tokens if k in ['input_ids', 'attention_mask']
            }
            outputs = self.model(**tokens)

        else:
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
            tokens = {'input_ids': input_ids}

            outputs = self.model(input_ids)
        return outputs.get('logits'), {'tokens': tokens}

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """

        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_ppl(inputs, mask_length=mask_length)
        else:
            return np.concatenate([
                self._get_ppl(inputs=[text], mask_length=mask_length)
                for text in inputs
            ])

    def _get_ppl(self,
                 inputs: List[str],
                 mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """

        outputs, inputs = self.get_logits(inputs)
        shift_logits = outputs[..., :-1, :].contiguous()

        shift_labels = inputs['tokens']['input_ids'][..., 1:].contiguous()

        self.tokenizer.pad_token_id = 0
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (inputs['tokens']['input_ids'] !=
                0).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        loss = loss.float()
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer.encode(prompt))

