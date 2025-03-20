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

# from transformers import LlamaForCausalLM
from transformers import AutoConfig
from transformers import AutoTokenizer
import datetime
import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# from .flash_utils_v2.modeling_internlm2_cached_flash_attn import InternLM2ForCausalLM
# from .flash_utils_v2.AttnCache import AttnCacheConfig

# from collie import CollieConfig

PromptType = Union[PromptList, str]

from ...huggingface import HuggingFaceCausalLM, BaseModel
from transformers.generation.utils import GenerationConfig

def print_gpu_memory_info():
    # 获取当前设备的显存信息
    current_device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(current_device).total_memory
    allocated_memory = torch.cuda.memory_allocated(current_device)
    cached_memory = torch.cuda.memory_cached(current_device)
    
    # 转换为GB单位
    total_memory_gb = total_memory / (1024 ** 3)
    allocated_memory_gb = allocated_memory / (1024 ** 3)
    cached_memory_gb = cached_memory / (1024 ** 3)
    
    # 计算使用百分比
    usage_percentage = (allocated_memory / total_memory) * 100
    
    print('*'*100)
    print(f"Device: {current_device}")
    print(f"Total Memory: {total_memory_gb:.2f} GB")
    print(f"Allocated Memory: {allocated_memory_gb:.2f} GB ({usage_percentage:.2f}% used)")
    print(f"Cached Memory: {cached_memory_gb:.2f} GB")
    print('*'*100)

@MODELS.register_module()
class LightCacheCausalLM(HuggingFaceCausalLM):
    def __init__(self,
                 path: str,
                 model_type: str,
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
                 attn_implementation: str = 'flash_attention_2', 
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
        self.model_type = model_type

        self._load_tokenizer(path=path, 
                             tokenizer_path=tokenizer_path,
                             tokenizer_kwargs=tokenizer_kwargs)
        
        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode
        if not tokenizer_only:
            self._load_model(path=path,
                             model_type=model_type, 
                             model_kwargs=model_kwargs,
                             peft_path=peft_path)
        self.generation_kwargs = generation_kwargs
        self.use_fastchat_template = use_fastchat_template
        self.end_str = end_str

        self.long_bench_cat = long_bench_cat
        self.prompt_format = prompt_format
                
        self.quanto_enable = quanto_enable
        self.chat_enable = chat_enable
        
        self.past_key_values = None
        self.generation_config = GenerationConfig(
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            num_beams=1, do_sample=False, use_cache=True
        )


    def _load_tokenizer(self, path: Optional[str], tokenizer_path: Optional[str], tokenizer_kwargs: dict):
        
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)  # , local_files_only=True

        super()._load_tokenizer(path=path, tokenizer_path=tokenizer_path, tokenizer_kwargs=tokenizer_kwargs)
        
        # if 'llama2' in tokenizer_path.lower() or 'llama-2' in tokenizer_path.lower():
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.eos_token_id
            # self.tokenizer.unk_token_id = self.tokenizer.bos_token_id

        # self.eos_token_id = self.tokenizer.eos_token_id
        # self.tokenizer.bos_token = '<s>'
        # if self.model_type in ['llama', 'internlm2', ]: 
        #     self.tokenizer.eos_token = '</s>'
        #     self.pad_token_id = self.tokenizer.pad_token_id
        # else:
        #     self.pad_token_id = self.tokenizer.eos_token_id

    def _load_model(self,
                    path: str, 
                    model_type: str, 
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        
        # config = CollieConfig.from_pretrained(path, trust_remote_code=True)  # , local_files_only=True

        # config.model_config.use_cache = True
        # config.model_config.attn_implementation = self.attn_implementation
        # config.checkpointing = True
        # config.use_flash = True
        # config.ds_config = {
        #     'bf16': {
        #         'enabled': True,
        #     },
        #     'train_micro_batch_size_per_gpu': 1, 
        #     'zero_optimization': {
        #         "stage": 3, 
        #     },            
        # }
        self.config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        self.config.attn_implementation = self.attn_implementation
        # self.long_cache_config['score_record'] = True
                
        if model_type == 'llama':
            from .cache_utils_v1030 import LightCacheConfig
            # from .cache_utils_v0921 import LightCacheConfig
            # from transformers import LlamaForCausalLM
            from .light_cache_llama_old import LlamaForCausalLM
            # assert type(self.long_cache_config) == dict, f"long_cache_config must be a dict, but got {type(self.long_cache_config)}"
            # print(self.long_cache_config, flush=True)
            self.long_cache_config = LightCacheConfig(num_key_value_heads=self.config.num_key_value_heads, 
                                                num_attention_heads=self.config.num_attention_heads,
                                                num_layers=self.config.num_hidden_layers,
                                                **self.long_cache_config)
            self.config.long_cache_config = self.long_cache_config
            self.config.rope_scaling = self.long_cache_config.rope_scaling
            print('self.config.rope_scaling', self.config.rope_scaling, flush=True)
            self.config.pretraining_tp = 1
            self._set_model_kwargs_torch_dtype(model_kwargs)
            self.model = LlamaForCausalLM.from_pretrained(path, **model_kwargs,
                                                          config=self.config, trust_remote_code=True,  # local_files_only=True, 
                                                          attn_implementation=self.attn_implementation)
        else:
            raise NotImplementedError(f"model_type {model_type} not supported")

        # elif model_type == 'llama+':

        #     from .cache_utils_v0921 import LightCacheConfig
        #     from .light_cache_llama3_1_ids import LlamaForCausalLM
        #     # assert type(self.long_cache_config) == dict, f"long_cache_config must be a dict, but got {type(self.long_cache_config)}"
        #     self.long_cache_config = LightCacheConfig(num_key_value_heads=self.config.num_key_value_heads, 
        #                                          num_attention_heads=self.config.num_attention_heads, 
        #                                          **self.long_cache_config)
        #     self.config.long_cache_config = self.long_cache_config
        #     print('self.config.rope_scaling', self.config.rope_scaling, flush=True)
        #     self.config.pretraining_tp = 1
        #     self._set_model_kwargs_torch_dtype(model_kwargs)
        #     self.model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto",  # name_dict,  # device,  # **model_kwargs,
        #                                                   config=self.config, trust_remote_code=True,  # local_files_only=True, 
        #                                                   attn_implementation=self.attn_implementation)
        #     # print(self.model.hf_device_map)

        # elif model_type == 'internlm2':
        #     from .cache_utils_v0921 import LightCacheConfig
        #     from .light_cache_internlm2 import InternLM2ForCausalLM
        #     # assert type(self.long_cache_config) == dict, f"long_cache_config must be a dict, but got {type(self.long_cache_config)}"
        #     self.long_cache_config = LightCacheConfig(num_key_value_heads=self.config.num_key_value_heads, 
        #                                          num_attention_heads=self.config.num_attention_heads, 
        #                                          **self.long_cache_config)
        #     self.config.long_cache_config = self.long_cache_config
        #     self.config.rope_scaling = self.long_cache_config.rope_scaling
        #     print('self.config.rope_scaling', self.config.rope_scaling, flush=True)
        #     self.config.pretraining_tp = 1
        #     self._set_model_kwargs_torch_dtype(model_kwargs)
        #     # self.config = config.model_config
            
        #     # from collie import setup_distribution
        #     # setup_distribution(config=config)
        #     # import torch.distributed as dist
        #     # dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        #     # # Set the device
        #     # device = torch.device(f"cuda:{int(os.environ['RANK'])}")
            
        #     self.model = InternLM2ForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto",  # device,  # **model_kwargs,
        #                                                       config=self.config, trust_remote_code=True,  # local_files_only=True, 
        #                                                       attn_implementation=self.attn_implementation)
        #     # # self.model.eval().cuda()
            
        #     # from torch.optim import AdamW
        #     # self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        #     # self.model.eval()
            
        # elif model_type == 'qwen2':
        #     from .cache_utils_v0921 import LightCacheConfig
        #     from .light_cache_qwen2 import Qwen2ForCausalLM
        #     # assert type(self.long_cache_config) == dict, f"long_cache_config must be a dict, but got {type(self.long_cache_config)}"
        #     self.long_cache_config = LightCacheConfig(num_key_value_heads=self.config.num_key_value_heads, 
        #                                          num_attention_heads=self.config.num_attention_heads, 
        #                                          **self.long_cache_config)
        #     self.config.long_cache_config = self.long_cache_config
        #     self.config.rope_scaling = self.long_cache_config.rope_scaling
        #     print('self.config.rope_scaling', self.config.rope_scaling, flush=True)
        #     self.config.pretraining_tp = 1
        #     self._set_model_kwargs_torch_dtype(model_kwargs)
        #     self.model = Qwen2ForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto",  # device,  # **model_kwargs,
        #                                                   config=self.config, trust_remote_code=True,  # local_files_only=True, 
        #                                                   attn_implementation=self.attn_implementation)

        # elif model_type == 'mistral':
        #     from .cache_utils_v0921 import LightCacheConfig
        #     from .light_cache_mistral import MistralForCausalLM
        #     # assert type(self.long_cache_config) == dict, f"long_cache_config must be a dict, but got {type(self.long_cache_config)}"
        #     self.long_cache_config = LightCacheConfig(num_key_value_heads=self.config.num_key_value_heads, 
        #                                          num_attention_heads=self.config.num_attention_heads, 
        #                                          **self.long_cache_config)
        #     self.config.long_cache_config = self.long_cache_config
        #     self.config.rope_scaling = self.long_cache_config.rope_scaling
        #     print('self.config.rope_scaling', self.config.rope_scaling, flush=True)
        #     self.config.pretraining_tp = 1
        #     self._set_model_kwargs_torch_dtype(model_kwargs)
        #     self.model = MistralForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto",  # device,  # **model_kwargs,
        #                                                   config=self.config, trust_remote_code=True,  # local_files_only=True, 
        #                                                   attn_implementation=self.attn_implementation)

        # elif model_type == 'chatglm':
        #     from .cache_utils import LightCacheConfig
        #     from .light_cache_chatglm import ChatGLMForConditionalGeneration
        #     # assert type(self.long_cache_config) == dict, f"long_cache_config must be a dict, but got {type(self.long_cache_config)}"
        #     self.long_cache_config = LightCacheConfig(num_key_value_heads=self.config.num_attention_heads // self.config.multi_query_group_num, 
        #                                          num_attention_heads=self.config.num_attention_heads, 
        #                                          **self.long_cache_config)
        #     self.config._attn_implementation = "flash_attention_2"
        #     self.config.long_cache_config = self.long_cache_config
        #     self.config.rope_scaling = self.long_cache_config.rope_scaling
        #     print('self.config.rope_scaling', self.config.rope_scaling, flush=True)
        #     self.config.pretraining_tp = 1
        #     self._set_model_kwargs_torch_dtype(model_kwargs)
        #     self.model = ChatGLMForConditionalGeneration.from_pretrained(
        #         path, torch_dtype=torch.float16, device_map="auto",  # device,  # **model_kwargs,
        #         config=self.config, trust_remote_code=True)

        # else:
        #     # self.config = config.model_config
        #     self.long_cache_config = {
        #         'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        #         'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        #         'key_compress_ratio': 1, 'value_compress_ratio': 1, }
        #     from .cache_utils import LightCacheConfig
        #     self.long_cache_config = LightCacheConfig(**self.long_cache_config)
        #     self.config.long_cache_config = self.long_cache_config
        #     from transformers import AutoModelForCausalLM
        #     self.config._flash_attn_2_enabled = True
        #     self.config.attn_implementation = "flash_attention_2"
        #     self.config._attn_implementation = "flash_attention_2"
        #     self._set_model_kwargs_torch_dtype(model_kwargs)
        #     self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs, config=self.config)
        
        # import ipdb
        # ipdb.set_trace()
        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                   peft_path,
                                                   is_trainable=False)
        self.model.eval()
        self.model.generation_config.do_sample = False

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]: 
        """Generate results given a list of inputs."""
        self.model.eval()  # Set the model to evaluation mode
        outputs_text = []
        
        # import sys
        
        # sys.path.append('/fs-computility/llm/shared/liuxiaoran/opencompass/opencompass/models/light_cache/')
        
        # import threading
        # from profiles import CUDAMemoryProfiler
                    
        # memory_profiler = CUDAMemoryProfiler(
        #     [self.model],
        #     filename='/fs-computility/llm/shared/liuxiaoran/opencompass/opencompass/models/light_cache/profile2.txt'
        # )

        # sys.settrace(memory_profiler)
        # threading.settrace(memory_profiler)   

        # from torch.profiler import profile, record_function, ProfilerActivity

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
             
        for text in inputs:
            
            if text.endswith('\n\n') and not self.chat_enable:
                text = text[:-2]
            
            if self.chat_enable:
                input_ids = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], 
                                                            tokenize=True, add_generation_prompt=True, 
                                                            return_tensors="pt")
            else:
                input_ids = self.tokenizer(text, return_tensors="pt").input_ids

            if self.long_bench_cat > 0:
                if input_ids.shape[-1] > self.long_bench_cat:
                    input_ids = torch.cat([input_ids[:, : self.long_bench_cat // 2], input_ids[:, - self.long_bench_cat // 2:]], dim=-1).to(device=self.model.device)
                else:
                    input_ids = input_ids.to(device=self.model.device)
            else:
                input_ids = input_ids.to(device=self.model.device)
            
            print(f"\n\ninput_ids.shape: {input_ids.shape}\n", flush=True)
            seq_len = input_ids.shape[-1]

            global_size = self.long_cache_config.global_size[0] if type(self.long_cache_config.global_size) == list else self.long_cache_config.global_size
            local_size = self.long_cache_config.local_size[0] if type(self.long_cache_config.local_size) == list else self.long_cache_config.local_size
            
            if self.long_cache_config.recall_option not in ['generate_only', 'full_attn', ]:            
                mod_len = (seq_len - 1 - global_size + local_size) % 128
                start, end = 0, global_size + local_size - (128 - mod_len)
                chunk_length = self.long_cache_config.chunk_size 
                # print(f'{type(self.past_key_values)}, {self.past_key_values is None}', flush=True)
                while start < seq_len - 1:
                    if start // 100000 != end // 100000:
                        print(datetime.datetime.now(), start, end, flush=True)
                    input_chunk = input_ids[:, start:min(end, seq_len - 1)]
                    outputs = self.model.forward(input_chunk, past_key_values=self.past_key_values)
                    start, end = end, end + chunk_length
                    self.past_key_values = outputs.past_key_values
                    torch.cuda.empty_cache()
                outputs = self.model.generate(input_ids, past_key_values=self.past_key_values, return_dict_in_generate=True, 
                                            max_new_tokens=max_out_len, generation_config=self.generation_config)
            # if self.long_cache_config.recall_option not in ['generate_only', 'full_attn', ] and seq_len > self.long_cache_config.first_prefill:
            #     mod_len = (seq_len - 1) % 128
            #     first_prefill = self.long_cache_config.first_prefill - (128 - mod_len)
            #     start, end = 0, first_prefill  # + self.long_cache_config.mid_size 
            #     chunk_length = self.long_cache_config.chunk_size
            #     # print(f'{type(self.past_key_values)}, {self.past_key_values is None}', flush=True)
            #     while start < seq_len - 1:
            #         if start // 100000 != end // 100000:
            #             print(datetime.datetime.now(), start, end, flush=True)
            #         input_chunk = input_ids[:, start:min(end, seq_len - 1)]
            #         outputs = self.model.forward(input_chunk, past_key_values=self.past_key_values)
            #         start, end = end, end + chunk_length
            #         self.past_key_values = outputs.past_key_values
            #         torch.cuda.empty_cache()
            #     outputs = self.model.generate(input_ids, past_key_values=self.past_key_values, return_dict_in_generate=True, 
            #                                 max_new_tokens=max_out_len, generation_config=self.generation_config)
            else:
                outputs = self.model.generate(input_ids, return_dict_in_generate=True, 
                                            max_new_tokens=max_out_len, generation_config=self.generation_config)
                
            self.past_key_values = outputs.past_key_values
            self.past_key_values = self.past_key_values.clean_cache()
            generated_text = self.tokenizer.decode(outputs.sequences[0, input_ids.shape[1]:], skip_special_tokens=True)

            outputs_text.append(generated_text)
            torch.cuda.empty_cache()
        
        # print(prof.table())
        # prof.export_chrome_trace('/fs-computility/llm/shared/liuxiaoran/opencompass/opencompass/models/light_cache/resnet_profile2.json')

        return outputs_text