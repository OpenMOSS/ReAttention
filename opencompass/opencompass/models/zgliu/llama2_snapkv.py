from typing import Optional
from opencompass.models import HuggingFaceCausalLM
from opencompass.registry import MODELS
import copy

from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    TextStreamer,
)
from typing import Tuple
import torch

@MODELS.register_module()
class Llama2_SnapKV(HuggingFaceCausalLM):
    def load_model(self, model_id, sparsity_method) -> Tuple[LlamaForCausalLM, LlamaTokenizer]:
        # model: MyLlamaForCausalLM
        config = AutoConfig.from_pretrained(model_id)
        if sparsity_method == "sink":
            self.logger.info("use_sink")
            config.use_sink = True
        elif sparsity_method == "h2o":
            self.logger.info("use_h2o")
            config.use_h2o = True
        elif sparsity_method == "snapkv":
            self.logger.info("use_snapkv")
            config.use_snapkv = True

        model = LlamaForCausalLM.from_pretrained(
            model_id,
            config=config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        return model
    
    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        from transformers import AutoModelForCausalLM

        self._set_model_kwargs_torch_dtype(model_kwargs)

        from .h2o_utils.modify_llama import hijack_llama
        hijack_llama()
        self.model = self.load_model(path, "snapkv")
        self.model.eval()
        self.model.generation_config.do_sample = False
        self.logger.info("Model loaded and converted with SnapKV method")