from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.infinitebench.infinitebenchenmc.infinitebench_enmc_gen import InfiniteBench_enmc_datasets
    from opencompass.configs.datasets.infinitebench.infinitebenchenqa.infinitebench_enqa_gen import InfiniteBench_enqa_datasets
    from opencompass.configs.datasets.infinitebench.infinitebenchensum.infinitebench_ensum_gen import InfiniteBench_ensum_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

from opencompass.models import Llama2_H2O
from opencompass.models import HuggingFaceBaseModel
import torch

models = [
    # dict(
    #     abbr='llama3_8B-h2o-8k', 
    #     type=Llama2_H2O,
    #     path='/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb/',
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16),
    #     tokenizer_path='/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb/',
    #     tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
    #     max_seq_len=7500,
    #     max_out_len=500,
    #     batch_size=1,
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    # ),
    dict(
        abbr='llama3_1_8B-h2o-32k', 
        type=Llama2_H2O,
        path='/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb/',
        model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16),
        tokenizer_path='/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb/',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        max_seq_len=31500,
        max_out_len=500,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ), 
    dict(
        abbr='llama3_1_8B-h2o-64k', 
        type=Llama2_H2O,
        path='/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb/',
        model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16),
        tokenizer_path='/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb/',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        max_seq_len=63500,
        max_out_len=500,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ), 
    dict(
        abbr='llama3_1_8B-h2o-128k', 
        type=Llama2_H2O,
        path='/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb/',
        model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16),
        tokenizer_path='/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb/',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        max_seq_len=127500,
        max_out_len=500,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ), 
]

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, DLCRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

work_dir = './outputs_xrliu/long_cache_bench/'

alillm2_workspace_id = "5360"
alillm2_resource_id = "quotaisboque9bap"
alillm2_data_sources = ("d-y3io773he2wcbc9pg9,d-gb3sr4nek0oo7g6t5o,d-3gpx15bjuemjldkh7f,d-0hbqqa80lbae52erw9,d-thsmec4dxu948ckeax,d-5hs894crzedhkc1eyi,d-mc0phdj9ek2jdk9cpl")

aliyun_cfg = dict(
    bashrc_path="/cpfs01/user/liuxiaoran/.bashrc",
    conda_env_name='/cpfs01/user/liuxiaoran/miniconda3/envs/llm-torch2.4', 
    # conda_env_name='/cpfs01/user/liuxiaoran/miniconda3/envs/llm-cuda12.1', 
    ali_submit_dlc_path='/nas/shared/public/songdemin/code/opencompass/run_ali_task.py',
    dlc_config_path="/cpfs01/user/liuxiaoran/dlc.config",
    workspace_id=alillm2_workspace_id,
    worker_image='dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/modelscope:1.16.1-pytorch2.3.0tensorflow2.16.1-gpu-py310-cu121-ubuntu22.04',
    resource_id=alillm2_resource_id,
    data_sources=alillm2_data_sources
)

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
    runner=dict(
        type=DLCRunner,
        max_num_workers=128,
        task=dict(type=OpenICLInferTask),
        aliyun_cfg=aliyun_cfg, 
        preemptible=True,  # False, #
        priority=6, 
        retry=1),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=DLCRunner,
        max_num_workers=64,
        task=dict(type=OpenICLEvalTask),
        aliyun_cfg=aliyun_cfg,
        preemptible=True,  # False, #
        priority=9, 
        retry=1),
)

# python run.py configs/eval_xrliu_h2o_infinitebench.py --dump-eval-details -r