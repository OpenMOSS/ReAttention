from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, DLCRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from .dataset_xrliu.long_score_long import datasets
    # from .model_xrliu.huggingface_32k import models
    from .model_xrliu.re_attention_32k import models
    # from .model_xrliu.re_attention_128k import models
    from .model_xrliu.vllm_128k import models

work_dir = './outputs_xrliu/long_cache_bench/'

alillm2_workspace_id = "5360"
alillm2_resource_id = "quotaisboque9bap"
alillm2_data_sources = ("d-y3io773he2wcbc9pg9,d-gb3sr4nek0oo7g6t5o,d-3gpx15bjuemjldkh7f,d-0hbqqa80lbae52erw9,d-thsmec4dxu948ckeax,d-5hs894crzedhkc1eyi,d-mc0phdj9ek2jdk9cpl")

aliyun_cfg = dict(
    bashrc_path="/cpfs01/user/liuxiaoran/.bashrc",
    # conda_env_name='/cpfs01/user/liuxiaoran/miniconda3/envs/llm-torch2.1', 
    conda_env_name='/cpfs01/user/liuxiaoran/miniconda3/envs/llm-torch2.4', 
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
        preemptible=False, 
        priority=6, 
        driver='535.54.03', 
        retry=4),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=DLCRunner,
        max_num_workers=64,
        task=dict(type=OpenICLEvalTask),
        aliyun_cfg=aliyun_cfg,
        preemptible=True, 
        priority=9, 
        retry=4),
)

# source /fs-computility/llm/liuxiaoran/.bashrc
# conda activate /cpfs01/user/liuxiaoran/miniconda3/envs/llm-torch2.1
# python run.py configs/eval_xrliu_re_attention_bench.py --dump-eval-details --debug -r 调试用
# python run.py configs/eval_xrliu_re_attention_bench.py --dump-eval-details -r 第一次用

# python run.py configs/eval_xrliu_re_attention_bench.py --dump-eval-details

#  . 第二次用
