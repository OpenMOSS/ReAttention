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
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from .rag_utils.rag_method import bm25_search, faiss_search, bm25_faiss_search, load_embedding_model, restore_sentence, mixed_segment

@MODELS.register_module()
class RAG_CONTEXT_LlamaForCausalLM(HuggingFaceCausalLM):
    def __init__(self, *args, **kwargs):
        self.rag_context_kwargs = kwargs.pop('rag_context_kwargs')
        super().__init__(*args, **kwargs)

    def generate(self,
                 inputs: List[PromptType],
                 max_out_len: int = 512,
                 skip_overlength=False,
                 **kwargs) -> str:
        
        if self.rag_context_kwargs is not None:
            rag_method_dict = {
                # 'bm25': bm25_search,
                # 'faiss': faiss_search,
                'bm25_faiss': bm25_faiss_search,
            }
            rag_method_func = rag_method_dict[self.rag_context_kwargs.get('rag_method')]
            embedding_model = load_embedding_model(self.rag_context_kwargs.get('embedding_model_path'))

        global_size = self.rag_context_kwargs.get('global_size', 32)
        local_size = self.rag_context_kwargs.get('local_size', 32)
        text_chunk_size = self.rag_context_kwargs.get('text_chunk_size', 64)
        text_chunk_overlap = self.rag_context_kwargs.get('text_chunk_overlap', 16)
        k = self.rag_context_kwargs.get('k', 16)
        
        new_inputs = []

        for input_text in inputs:
            assert isinstance(input_text, str)
            
            tokenized_text = mixed_segment(input_text)
            
            tokenized_chunk_text = [tokenized_text[i:i + text_chunk_size] for i in range(0, len(tokenized_text), text_chunk_size - text_chunk_overlap)]

            global_part = tokenized_text[:global_size]
            local_part = tokenized_text[-local_size:]

            query_list = [global_part, local_part]

            result = rag_method_func(embedding_model, tokenized_chunk_text, query_list, k)
            selected_chunk = [tokenized_chunk_text[idx] for idx in result]
            
            global_part_text = restore_sentence(global_part)
            local_part_text = restore_sentence(local_part)
            selected_chunk_text = ' '.join([restore_sentence(chunk) for chunk in selected_chunk])
            
            new_input_text = "Context: " + selected_chunk_text + '\n\n' + global_part_text + '\n\n' + local_part_text
            new_input_text = global_part_text + '\n\n' + "Context: " + selected_chunk_text + '\n\n' + local_part_text
            
            new_inputs.append(new_input_text)
        
            old_input_length = len(tokenized_text)
            new_input_length = len(mixed_segment(new_input_text))
            # print(f"Input length: {old_input_length} -> {new_input_length}")
            # print(f"{input_text=}")
            # print(f"{new_input_text=}")
            
        
        return super().generate(new_inputs, max_out_len, skip_overlength, **kwargs)