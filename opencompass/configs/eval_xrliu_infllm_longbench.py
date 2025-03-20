import torch
from mmengine.config import read_base

with read_base():

    from opencompass.configs.datasets.longbench.longbenchnarrativeqa.longbench_narrativeqa_gen import LongBench_narrativeqa_datasets
    from opencompass.configs.datasets.longbench.longbenchqasper.longbench_qasper_gen import LongBench_qasper_datasets
    from opencompass.configs.datasets.longbench.longbenchmultifieldqa_en.longbench_multifieldqa_en_gen import LongBench_multifieldqa_en_datasets
    from opencompass.configs.datasets.longbench.longbenchmultifieldqa_zh.longbench_multifieldqa_zh_gen import LongBench_multifieldqa_zh_datasets

    from opencompass.configs.datasets.longbench.longbenchhotpotqa.longbench_hotpotqa_gen import LongBench_hotpotqa_datasets
    from opencompass.configs.datasets.longbench.longbench2wikimqa.longbench_2wikimqa_gen import LongBench_2wikimqa_datasets
    from opencompass.configs.datasets.longbench.longbenchmusique.longbench_musique_gen import LongBench_musique_datasets
    from opencompass.configs.datasets.longbench.longbenchdureader.longbench_dureader_gen import LongBench_dureader_datasets

    from opencompass.configs.datasets.longbench.longbenchgov_report.longbench_gov_report_gen import LongBench_gov_report_datasets
    from opencompass.configs.datasets.longbench.longbenchqmsum.longbench_qmsum_gen import LongBench_qmsum_datasets
    from opencompass.configs.datasets.longbench.longbenchmulti_news.longbench_multi_news_gen import LongBench_multi_news_datasets
    from opencompass.configs.datasets.longbench.longbenchvcsum.longbench_vcsum_gen import LongBench_vcsum_datasets

    from opencompass.configs.datasets.longbench.longbenchtrec.longbench_trec_gen import LongBench_trec_datasets
    from opencompass.configs.datasets.longbench.longbenchtriviaqa.longbench_triviaqa_gen import LongBench_triviaqa_datasets
    from opencompass.configs.datasets.longbench.longbenchsamsum.longbench_samsum_gen import LongBench_samsum_datasets
    from opencompass.configs.datasets.longbench.longbenchlsht.longbench_lsht_gen import LongBench_lsht_datasets

    from opencompass.configs.datasets.longbench.longbenchpassage_count.longbench_passage_count_gen import LongBench_passage_count_datasets
    from opencompass.configs.datasets.longbench.longbenchpassage_retrieval_en.longbench_passage_retrieval_en_gen import LongBench_passage_retrieval_en_datasets
    from opencompass.configs.datasets.longbench.longbenchpassage_retrieval_zh.longbench_passage_retrieval_zh_gen import LongBench_passage_retrieval_zh_datasets

    from opencompass.configs.datasets.longbench.longbenchlcc.longbench_lcc_gen import LongBench_lcc_datasets
    from opencompass.configs.datasets.longbench.longbenchrepobench.longbench_repobench_gen import LongBench_repobench_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

path_dict = {'llama2_7B': '', 
             'llama2_7B_chat': '', 
             'llama2_13B': '', 
             'llama3_8B': '/cpfs01/shared/alillm2/llm_llama3_hf/Meta-Llama-3-8B/', 
             'llama3_8B_chat': '/cpfs01/shared/alillm2/llm_llama3_hf/Meta-Llama-3-8B-Instruct/', 
             'llama3_1_8B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb/', 
             'llama3_1_8B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/07eb05b21d191a58c577b4a45982fe0c049d0693/', 
             'llama3_1_70B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-70B/snapshots/7740ff69081bd553f4879f71eebcc2d6df2fbcb3/', 
             'llama3_1_70B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/846357c7ee5e3f50575fd4294edb3d898c8ea100/', 
             'llama3_2_3B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/392a143b624368100f77a3eafaa4a2468ba50a72/', 
             'internlm2_1B': '',
             'internlm2_1B_base': '',
             'internlm2_7B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--internlm--internlm2-7b/snapshots/fa732f7cd8ec6299400e37e790ecd82abb1a7f9a/',
             'internlm2_7B_base': '',  
             'internlm2_7B_chat': '',
             'internlm2_20B': '',
             'internlm2.5_7B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--internlm--internlm2_5-7b/snapshots/7bee5c5d7c4591d7f081f5976610195fdd1f1e35/',  # '/nas/shared/alillm2/zhangshuo/exported_transformers/official_Ampere2.5_7B_Enhance_2.0.0/50000/', 
             'internlm2.5_7B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--internlm--internlm2_5-7b-chat-1m/snapshots/8d1a709a04d71440ef3df6ebbe204672f411c8b6/', 

             'qwen1_5_14B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen1.5-14B/snapshots/dce4b190d34470818e5bec2a92cb8233aaa02ca2/', 
             'qwen1_5_32B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen1.5-32B/snapshots/cefef80dc06a65f89d1d71d0adbc56d335ca2490/', 
             'qwen2_1B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen2-1.5B/snapshots/8a16abf2848eda07cc5253dec660bf1ce007ad7a/', 
             'qwen2_7B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen2-7B/snapshots/d8412313bc00677839b4e38cdc751d536ccc12ab/', 
             'qwen2_72B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen2-72B/snapshots/e5cebedf0244946b1ad5eb2d753ca3c1a90f11fb/', 
             'mistral3_7B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--mistralai--Mistral-7B-v0.3/snapshots/b67d6a03ca097c5122fa65904fce0413500bf8c8/', 
             'mistral3_7B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de/', 
             'glm4_9B_chat_1M': '/nas/shared/public/llmeval/model_weights/hf_hub/models--THUDM--glm-4-9b-chat-1m/snapshots/715ddbe91082f976ff6a4ca06d59e5bbff6c3642/', 

             'moss2_13B_v2_4000': '/nas/shared/public/liuxiaoran/tmp_ckpts_hf/moss2-13b-v2/4000/', 
            }

from opencompass.models.xrliu.infllm_model import InfLLM_xrliu

tags = [
    # ('-infllm-32k_cat', 'llama3_8B_chat', 31500, 1, 
    #  dict(model_center=False, type='inf-llm', block_size=128, fattn=False, 
    #       n_init=128, n_local=4096, topk=16, repr_topk=4, max_cached_block=32, exc_block_size=512, chunk_size=8192)),  

    # ('-infllm-128.31x128.4096.512-32k_cat', 'llama3_8B', 31500, 1, 
    #  dict(model_center=False, type='inf-llm', block_size=128, fattn=False, 
    #       n_init=128, n_local=4096, topk=31, repr_topk=4, max_cached_block=32, exc_block_size=512, chunk_size=512)), 
    # ('-infllm-128.31x128.4096.512-32k_cat', 'llama3_1_8B', 31500, 1, 
    #  dict(model_center=False, type='inf-llm', block_size=128, fattn=False, 
    #       n_init=128, n_local=4096, topk=31, repr_topk=4, max_cached_block=32, exc_block_size=512, chunk_size=512)), 
    # ('-infllm-128.31x128.4096.512-32k_cat', 'llama3_2_3B_chat', 31500, 1, 
    #  dict(model_center=False, type='inf-llm', block_size=128, fattn=False, 
    #       n_init=128, n_local=4096, topk=31, repr_topk=4, max_cached_block=32, exc_block_size=512, chunk_size=512)), 

    # ('-infllm-32k_cat', 'llama3_8B', 31500, 1, 
    #  dict(model_center=False, type='inf-llm', block_size=128, fattn=False, 
    #       n_init=128, n_local=4096, topk=16, repr_topk=4, max_cached_block=32, exc_block_size=512, chunk_size=8192)),  
    ('-infllm-32.127x32.4096.512-32k_cat', 'llama3_8B', 31500, 1, 
     dict(model_center=False, type='inf-llm', block_size=32, fattn=False, 
          n_init=32, n_local=4096, topk=127, repr_topk=4, max_cached_block=128, exc_block_size=512, chunk_size=512)), 
    # ('-infllm-32.255x32.8192.512-32k_cat', 'llama3_1_8B', 31500, 1, 
    #  dict(model_center=False, type='inf-llm', block_size=32, fattn=False, 
    #       n_init=32, n_local=8192, topk=255, repr_topk=4, max_cached_block=256, exc_block_size=512, chunk_size=512)), 
    # ('-infllm-32.255x32.8192.512-32k_cat', 'llama3_2_3B_chat', 31500, 1, 
    #  dict(model_center=False, type='inf-llm', block_size=32, fattn=False, 
    #       n_init=32, n_local=8192, topk=255, repr_topk=4, max_cached_block=256, exc_block_size=512, chunk_size=512)), 
]

models = []

for abbr, name, max_len, num_gpus, infllm_kwargs in tags:
    
    models.append(
        dict(
            abbr=f'{name}{abbr}', 
            type=InfLLM_xrliu,
            path=path_dict[name],
            model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16),
            infllm_kwargs=infllm_kwargs,
            tokenizer_path=path_dict[name],
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
            max_seq_len=max_len,
            max_out_len=500,
            run_cfg=dict(num_gpus=num_gpus, num_procs=1),
            batch_size=1,
        ),
    )

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, DLCRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

work_dir = './outputs_xrliu/long_cache_bench/'

alillm2_workspace_id = "5360"
alillm2_resource_id = "quotaisboque9bap"
alillm2_data_sources = ("d-y3io773he2wcbc9pg9,d-gb3sr4nek0oo7g6t5o,d-3gpx15bjuemjldkh7f,d-0hbqqa80lbae52erw9,d-thsmec4dxu948ckeax,d-5hs894crzedhkc1eyi,d-mc0phdj9ek2jdk9cpl")

aliyun_cfg = dict(
    bashrc_path="/cpfs01/user/liuxiaoran/.bashrc",
    conda_env_name='/cpfs01/user/liuxiaoran/miniconda3/envs/llm-shearing', 
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
        retry=4),
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
        retry=4),
)

# python run.py configs/eval_xrliu_infllm_longbench.py --dump-eval-details -r