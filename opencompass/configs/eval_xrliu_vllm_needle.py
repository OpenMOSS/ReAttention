from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, DLCRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():    
    from .model_xrliu.vllm_needle import models
    from .datasets.needlebench.needlebench.needlebench_single import needlebench_en_datasets as needlebench_origin_en_datasets

    # from .model_xrliu.re_attention_needle_multi import models
    # from .datasets.needlebench.needlebench.needlebench_multi_retrieval import needlebench_en_datasets as needlebench_parallel_en_datasets

    # from .datasets.needlebench.needlebench.needlebench_single import needlebench_zh_datasets as needlebench_origin_zh_datasets
    # from .datasets.needlebench.needlebench.needlebench_multi_retrieval import needlebench_zh_datasets as needlebench_parallel_zh_datasets

    # from .model_xrliu.turbomind_needle import models

    from .summarizers.needlebench import needlebench_summarizer as summarizer

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

work_dir = './outputs_xrliu/vllm_needle/'

alillm2_workspace_id = "5360"
alillm2_resource_id = "quotaisboque9bap"
alillm2_data_sources = ("d-y3io773he2wcbc9pg9,d-gb3sr4nek0oo7g6t5o,d-3gpx15bjuemjldkh7f,d-0hbqqa80lbae52erw9,d-thsmec4dxu948ckeax,d-5hs894crzedhkc1eyi,d-mc0phdj9ek2jdk9cpl")

aliyun_cfg = dict(
    bashrc_path="/cpfs01/user/liuxiaoran/.bashrc",
    conda_env_name='/cpfs01/user/liuxiaoran/miniconda3/envs/llm-torch2.4', 
    ali_submit_dlc_path='/nas/shared/public/songdemin/code/opencompass/run_ali_task.py',
    dlc_config_path="/cpfs01/user/liuxiaoran/dlc.config",
    workspace_id=alillm2_workspace_id,
    worker_image='dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/modelscope:1.16.1-pytorch2.3.0tensorflow2.16.1-gpu-py310-cu121-ubuntu22.04',
    resource_id=alillm2_resource_id,
    data_sources=alillm2_data_sources
)

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=DLCRunner,
        max_num_workers=88, 
        task=dict(type=OpenICLInferTask), 
        aliyun_cfg=aliyun_cfg,
        preemptible=False,  # True,  # 
        priority=6, 
        retry=2),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=DLCRunner,
        max_num_workers=88,
        task=dict(type=OpenICLEvalTask),
        aliyun_cfg=aliyun_cfg,
        preemptible=False,  # True,  # 
        priority=6,  # 9,  # 
        retry=2),
)

# source /fs-computility/llm/liuxiaoran/.bashrc
# conda activate /cpfs01/user/liuxiaoran/miniconda3/envs/llm-cuda12.1
# python run.py configs/eval_xrliu_vllm_needle.py --dump-eval-details --debug -r 调试用
# python run.py configs/eval_xrliu_vllm_needle.py --dump-eval-details -r 第一次用
#  . 第二次用
