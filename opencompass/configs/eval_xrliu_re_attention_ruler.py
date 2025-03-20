from opencompass.partitioners import (
    NaivePartitioner,
    NumWorkerPartitioner,
)
from mmengine.config import read_base
from opencompass.runners import LocalRunner, DLCRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from .datasets.ruler.ruler_combined_gen import ruler_combined_datasets
    # from ..configs.summarizers.groups.ruler import ruler_summary_groups

    # from .model_xrliu.re_attention_needle import models
    from .model_xrliu.turbomind_needle import models

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

work_dir = './outputs_xrliu/long_cache_ruler'

alillm2_workspace_id = "5360"
alillm2_resource_id = "quotaisboque9bap"
alillm2_data_sources = ("d-y3io773he2wcbc9pg9,d-gb3sr4nek0oo7g6t5o,d-3gpx15bjuemjldkh7f,d-0hbqqa80lbae52erw9,d-thsmec4dxu948ckeax,d-5hs894crzedhkc1eyi,d-mc0phdj9ek2jdk9cpl")

aliyun_cfg = dict(
    bashrc_path="/cpfs01/user/liuxiaoran/.bashrc",
    conda_env_name='/cpfs01/shared/public/changcheng/environment/oc-v035-ld-v062_yoco/', 
    # conda_env_name='/cpfs01/user/liuxiaoran/miniconda3/envs/llm-lmdeploy', 
    # conda_env_name='/cpfs01/user/liuxiaoran/miniconda3/envs/llm-torch2.1', 
    ali_submit_dlc_path='/nas/shared/public/songdemin/code/opencompass/run_ali_task.py',
    dlc_config_path="/cpfs01/user/liuxiaoran/dlc.config",
    workspace_id=alillm2_workspace_id,
    worker_image='dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/modelscope:1.16.1-pytorch2.3.0tensorflow2.16.1-gpu-py310-cu121-ubuntu22.04',
    resource_id=alillm2_resource_id,
    data_sources=alillm2_data_sources
)

# from opencompass.models import HuggingFaceBaseModel
# from opencompass.models import TurboMindModelLong

# models = [
#     dict(
#         type=TurboMindModelLong,
#         engine_config=dict(session_len=128000, max_batch_size=1),
#         gen_config=dict(top_k=1, top_p=1, temperature=1.0, max_new_tokens=500),
#         max_out_len=50,
#         max_seq_len=131072,
#         batch_size=1,
#         concurrency=1,
#         run_cfg=dict(num_gpus=1, num_procs=1),
#         end_str='<eoa>',
#         )
#     # dict(
#     #     type=HuggingFaceBaseModel,
#     #     abbr='internlm2.5_7B_chat',
#     #     path='/nas/shared/public/llmeval/model_weights/hf_hub/models--internlm--internlm2_5-7b-chat-1m/snapshots/8d1a709a04d71440ef3df6ebbe204672f411c8b6/', 
#     #     model_kwargs=dict(torch_dtype='float16', device_map='auto', 
#     #                       trust_remote_code=True, attn_implementation='flash_attention_2'), 
#     #     tokenizer_kwargs=dict(padding_side='left', truncation_side='left',
#     #                           trust_remote_code=True, use_fast=False, ),
#     #     max_out_len=50,
#     #     batch_size=1,
#     #     run_cfg=dict(num_gpus=1, 
#     #                  num_procs=1), 
#     # )
# ]

infer = dict(
    partitioner=dict(type=NaivePartitioner),  # dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(
        type=DLCRunner,
        max_num_workers=256,  # 84,
        task=dict(type=OpenICLInferTask),
        aliyun_cfg=aliyun_cfg, 
        preemptible=True, 
        priority=6, 
        retry=2),
)


eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=DLCRunner,
        max_num_workers=84,
        task=dict(type=OpenICLEvalTask),
        aliyun_cfg=aliyun_cfg,
        preemptible=True, 
        priority=9, 
        retry=2),
)


# summarizer = dict(
#     dataset_abbrs=['ruler_4k', 'ruler_8k', 'ruler_16k', 'ruler_32k', 'ruler_128k'],
#     summary_groups=sum(
#         [v for k, v in locals().items() if k.endswith('_summary_groups')], []
#     ),
# )

# source /fs-computility/llm/liuxiaoran/.bashrc
# conda activate /cpfs01/user/liuxiaoran/miniconda3/envs/llm-cuda12.1
# python run.py configs/eval_xrliu_re_attention_ruler.py --dump-eval-details --debug -r  调试用
# python run.py configs/eval_xrliu_re_attention_ruler.py --dump-eval-details -r 20240820_190019 第一次用
#  . 第二次用

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# dataset    version    metric         mode      qwen2-7b-instruct-turbomind    llama-3-8b-instruct-turbomind    internlm2_5-7b-chat-1m-turbomind
# ---------  ---------  -------------  ------  -----------------------------  -------------------------------  ----------------------------------
# 4k         -          naive_average  gen                             93.66                            93.48                               91.20
# 8k         -          naive_average  gen                             88.38                            89.95                               89.07
# 16k        -          naive_average  gen                             84.27                             0.14                               87.61
# 32k        -          naive_average  gen                             81.36                             0.00                               84.59
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
