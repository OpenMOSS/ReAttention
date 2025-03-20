from mmengine.config import read_base
from opencompass.models import TurboMindModelLong

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # choose a list of datasets
    from .datasets.collections.C_long import datasets
    # and output the results in a choosen format

work_dir = './outputs/flash_kv_cache/'

path_dict = {'llama2_7B': '/fs-computility/llm/shared/llm_data/llm_llama/llama2/llama-2-7b-hf/', 
             'llama2_7B_chat': '/fs-computility/llm/shared/llm_data/llm_llama/llama2/llama-2-7b-chat-hf/', 
             'llama2_13B': '/fs-computility/llm/shared/llm_data/llm_llama/llama2/llama-2-13b-hf/', 
             'llama3_8B': '/fs-computility/llm/shared/llm_llama3_hf/Meta-Llama-3-8B/', 
             'llama3_8B_chat': '/fs-computility/llm/shared/llm_llama3_hf/Meta-Llama-3-8B-Instruct/', 
             'internlm2_1B': '/fs-computility/llm/shared/zhangshuo/exported_transformers/official_Luyou_1B_2.3.0_Enhance_1.3.0_fix_base_fix_watermark_fix_famous/50000/',
             'internlm2_1B_base': '/fs-computility/llm/shared/zhangshuo/exported_transformers/official_Luyou_1B_2.3.0/464000/',
             'internlm2_7B': '/fs-computility/llm/shared/zhangshuo/exported_transformers/official_Ampere_7B_Enhance_1.2.0rc/50000/',
             'internlm2_7B_base': '/fs-computility/llm/shared/zhangshuo/exported_transformers/official_Ampere_7B_4_9_0_fix_watermark_add_famous/402000/',  
             'qwen1.5_1B': '/fs-computility/llm/shared/models/Qwen1.5-1.8B/', 
             'qwen1.5_7B': '/fs-computility/llm/shared/liuxiaoran/models--Qwen--Qwen1.5-7B/', 
             'huawei_moss': '/fs-computility/llm/shared/liuxiaoran/tmp_ckpts_hf/long_score-internlm2_7B-b1000000_0127/2000/',  
            }

num_gpus = {'llama2_7B': 1, 'llama2_7B_chat': 1, 'llama2_13B': 2, 'llama3_8B': 1, 'llama3_8B_chat': 1, 
            'qwen1.5_1B': 1, 'qwen1.5_7B': 1, 
            'internlm2_7B': 1, 'internlm2_7B_base': 1, 'internlm2_1B': 1, 'internlm2_1B_base': 1, 'huawei_moss': 1}

tags = [
        # ('-full_attn-q_int8-32k_cat', 'internlm2_7B', 31500, 
        #  dict(session_len=32000, max_batch_size=1, quant_policy=8)), 
        # ('-full_attn-q_int8-32k_cat', 'internlm2_1B', 31500, 
        #  dict(session_len=32000, max_batch_size=1, quant_policy=8)), 
        # ('-full_attn-q_int8-32k_cat', 'qwen1.5_1B', 31500, 
        #  dict(session_len=32000, max_batch_size=1, quant_policy=8)), 
        # ('-full_attn-q_int8-32k_cat', 'qwen1.5_7B', 31500, 
        #  dict(session_len=32000, max_batch_size=1, quant_policy=8)), 
        
        ('-full_attn-q_int4-32k_cat', 'internlm2_7B', 31500, 
         dict(session_len=32000, max_batch_size=1, quant_policy=4)), 
        ('-full_attn-q_int4-32k_cat', 'internlm2_1B', 31500, 
         dict(session_len=32000, max_batch_size=1, quant_policy=4)), 
        ('-full_attn-q_int4-32k_cat', 'qwen1.5_1B', 31500, 
         dict(session_len=32000, max_batch_size=1, quant_policy=4)), 
        ('-full_attn-q_int4-32k_cat', 'qwen1.5_7B', 31500, 
         dict(session_len=32000, max_batch_size=1, quant_policy=4)), 
    ]

models = []

for abbr, group, cat_len, engine_config in tags:
    models.append(
        dict(
            type=TurboMindModelLong,
            abbr=f'{group}{abbr}',
            path=path_dict[group],
            engine_config=engine_config,
            gen_config=dict(top_k=1, top_p=1, temperature=1.0, max_new_tokens=500),
            max_out_len=500,
            max_seq_len=32000,
            batch_size=1,
            concurrency=1,
            run_cfg=dict(num_gpus=num_gpus[group.split('-')[0]], 
                         num_procs=num_gpus[group.split('-')[0]]),
            end_str='<eoa>',
        )
    )

volcano_infer_cfg = dict(
    bashrc_path="/fs-computility/llm/liuxiaoran/.bashrc",  # bashrc 路径
    # conda_env_name='flash2.0',  #conda 环境名
    conda_path="/fs-computility/llm/liuxiaoran/miniconda3/envs/transformers",  #conda环境启动路径
    volcano_config_path="/fs-computility/llm/shared/liuxiaoran/opencompass/configs/configs/volc_config/volcano_infer.yaml"  #配置文件路径
)

volcano_eval_cfg = dict(
    bashrc_path="/fs-computility/llm/liuxiaoran/.bashrc",  # bashrc 路径
    # conda_env_name='flash2.0',
    conda_path="/fs-computility/llm/liuxiaoran/miniconda3/envs/transformers",
    volcano_config_path="/fs-computility/llm/shared/liuxiaoran/opencompass/configs/configs/volc_config/volcano_eval.yaml"
)

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
    runner=dict(
        type=VOLCRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask), 
        volcano_cfg=volcano_infer_cfg, 
        queue_name='hsllm_c', 
        # preemptible=True, 
        retry=2),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=VOLCRunner,
        max_num_workers=64,
        task=dict(type=OpenICLEvalTask),
        volcano_cfg=volcano_eval_cfg, 
        queue_name='hsllm_c', 
        # preemptible=True, 
        retry=2),
)
