from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import ReAttentionCausalLM

import torch

with read_base():

    ## infinitebench
    
    from opencompass.configs.datasets.infinitebench.infinitebenchenmc.infinitebench_enmc_gen import InfiniteBench_enmc_datasets
    from opencompass.configs.datasets.infinitebench.infinitebenchenqa.infinitebench_enqa_gen import InfiniteBench_enqa_datasets
    from opencompass.configs.datasets.infinitebench.infinitebenchensum.infinitebench_ensum_gen import InfiniteBench_ensum_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

num_gpus = {
    'llama3_8b': 1, 'llama3_8b_chat': 1, 'mistral3_7b': 1, 

    'llama3_1_8b': 1, 'llama3_1_8b_chat': 1, 'llama3_2_3b': 1, 'llama3_2_3b_chat': 1, 

    'qwen2_5_7b': 1, 'qwen2_5_7b_chat': 1, 'qwen2_5_1b': 1, 'qwen2_5_1b_chat': 1, 

    'internlm2_5_7b': 1, 
}

path_dict = {
    'llama3_8b': 'meta-llama/Meta-Llama-3-8B', 
    'llama3_8b_chat': 'meta-llama/Meta-Llama-3-8B-Instruct', 

    'llama3_1_8b': 'meta-llama/Llama-3.1-8B',
    'llama3_1_8b_chat': 'meta-llama/Llama-3.1-8B-Instruct', 
    'llama3_2_3b': 'meta-llama/Llama-3.2-3B',
    'llama3_2_3b_chat': 'meta-llama/Llama-3.2-3B-Instruct',

    'mistral3_7b': 'mistralai/Mistral-7B-v0.3', 

    'internlm2_5_7b': 'internlm/internlm2_5-7b', 

    'qwen2_5_7b': 'Qwen/Qwen2.5-7B', 
    'qwen2_5_7b_chat': 'Qwen/Qwen2.5-7B-Instruct', 
    'qwen2_5_1b': 'Qwen/Qwen2.5-1.5B', 
    'qwen2_5_1b_chat': 'Qwen/Qwen2.5-1.5B-Instruct', 
}

tags = [

    ## main results: 

    ### llama3_8b_chat

    ('-slm-32.0.4096.512-32k_cat', 'llama3_8b_chat', 'llama', '{prompt}', 31500,  
        {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, }),
    ('-slm-32.0.4096.512-64k_cat', 'llama3_8b_chat', 'llama', '{prompt}', 63500,  
        {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, }),
    ('-slm-32.0.4096.512-128k_cat', 'llama3_8b_chat', 'llama', '{prompt}', 127500,  
        {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, }),

    ('-reattn-32.4x32.4096.512-gu-clip127-32k_cat', 'llama3_8b_chat', 'llama', '{prompt}', 31500,  
        {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, }),
    ('-reattn-32.4x32.4096.512-gu-clip127-64k_cat', 'llama3_8b_chat', 'llama', '{prompt}', 63500,  
        {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, }),
    ('-reattn-32.4x32.4096.512-gu-clip127-128k_cat', 'llama3_8b_chat', 'llama', '{prompt}', 127500,  
        {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, }),

    ### llama3_1_8b_chat

    ('-slm-32.0.8192.512-32k_cat', 'llama3_1_8b_chat', 'llama+', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
    ('-slm-32.0.8192.512-64k_cat', 'llama3_1_8b_chat', 'llama+', '{prompt}', 63500, 
        {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
    ('-slm-32.0.8192.512-128k_cat', 'llama3_1_8b_chat', 'llama+', '{prompt}', 127500, 
        {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
    
    ('-reattn-32.4x32.8192.512-gu-clip256-32k_cat', 'llama3_1_8b_chat', 'llama+', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
    ('-reattn-32.4x32.8192.512-gu-clip256-64k_cat', 'llama3_1_8b_chat', 'llama+', '{prompt}', 63500, 
        {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
    ('-reattn-32.4x32.8192.512-gu-clip256-128k_cat', 'llama3_1_8b_chat', 'llama+', '{prompt}', 127500, 
        {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 

    ### llama3_2_3b_chat

    ('-slm-32.0.8192.512-32k_cat', 'llama3_2_3b_chat', 'llama+', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
    ('-slm-32.0.8192.512-64k_cat', 'llama3_2_3b_chat', 'llama+', '{prompt}', 63500, 
        {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
    ('-slm-32.0.8192.512-128k_cat', 'llama3_2_3b_chat', 'llama+', '{prompt}', 127500, 
        {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 

    ('-reattn-32.4x32.8192.512-gu-clip256-32k_cat', 'llama3_2_3b_chat', 'llama+', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
    ('-reattn-32.4x32.8192.512-gu-clip256-64k_cat', 'llama3_2_3b_chat', 'llama+', '{prompt}', 63500, 
        {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
    ('-reattn-32.4x32.8192.512-gu-clip256-128k_cat', 'llama3_2_3b_chat', 'llama+', '{prompt}', 127500, 
        {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 

]

models = []

for abbr, group, model_type, prompt_format, long_bench_cat, re_attn_config in tags:

    model_dict = {'attn_implementation': 'flash_attention_2'}
    model_dict.update(
        dict(
            type=ReAttentionCausalLM, 
            abbr=f'{group}{abbr}',
            path=path_dict[group],
            model_type=model_type,
            tokenizer_path=path_dict[group],
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', 
                                  use_fast=False, trust_remote_code=True), 
            max_out_len=50,
            batch_size=1, 
            model_kwargs=dict(device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True),
            re_attn_config=re_attn_config,
            long_bench_cat=long_bench_cat,
            prompt_format=prompt_format, 
            batch_padding=False, # if false, inference with for-loop without batch padding
            run_cfg=dict(num_gpus=num_gpus[group.split('-')[0]], #  1, 
                         num_procs=1), 
            chat_enable=False,
        )
    )

    models.append(model_dict)

work_dir = './outputs/reattn_infinite/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)

# python run.py eval_xrliu/eval_reattn_infinite.py --dump-eval-details -r --debug
# python run.py eval_xrliu/eval_reattn_infinite.py --dump-eval-details -r
