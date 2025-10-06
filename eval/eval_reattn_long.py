from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import ReAttentionCausalLM

import torch

with read_base():

    ## longbench

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

    ## leval
    
    from opencompass.configs.datasets.leval.levaltpo.leval_tpo_gen import LEval_tpo_datasets
    from opencompass.configs.datasets.leval.levalgsm100.leval_gsm100_gen import LEval_gsm100_datasets
    from opencompass.configs.datasets.leval.levalquality.leval_quality_gen import LEval_quality_datasets
    from opencompass.configs.datasets.leval.levalcoursera.leval_coursera_gen import LEval_coursera_datasets
    from opencompass.configs.datasets.leval.levaltopicretrieval.leval_topic_retrieval_gen import LEval_tr_datasets
    from opencompass.configs.datasets.leval.levalscientificqa.leval_scientificqa_gen import LEval_scientificqa_datasets
    
    from opencompass.configs.datasets.leval.levalmultidocqa.leval_multidocqa_gen import LEval_multidocqa_datasets
    from opencompass.configs.datasets.leval.levalpaperassistant.leval_paper_assistant_gen import LEval_ps_summ_datasets
    from opencompass.configs.datasets.leval.levalnaturalquestion.leval_naturalquestion_gen import LEval_nq_datasets
    from opencompass.configs.datasets.leval.levalfinancialqa.leval_financialqa_gen import LEval_financialqa_datasets
    from opencompass.configs.datasets.leval.levallegalcontractqa.leval_legalcontractqa_gen import LEval_legalqa_datasets
    from opencompass.configs.datasets.leval.levalnarrativeqa.leval_narrativeqa_gen import LEval_narrativeqa_datasets

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

    ### llama3_8b

    ('-8k_cat', 'llama3_8b', 'llama', '{prompt}', 7500, 
        {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
    ('-ntk-d4-32k_cat', 'llama3_8b', 'llama', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        'rope_scaling': {'type': 'dynamic', 'factor': 4, }, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
    ('-slm-32.0.4096.512-32k_cat', 'llama3_8b', 'llama', '{prompt}', 31500,  
        {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, }),
    # ('-reattn-32.4x32.4096.512-hu-clip127-32k_cat', 'llama3_8b', 'llama', '{prompt}', 31500,  
    #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
    #     'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
    #     'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
    #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, }),
    ('-reattn-32.4x32.4096.512-gu-clip127-32k_cat', 'llama3_8b', 'llama', '{prompt}', 31500,  
        {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, }),

    ### llama3_1_8b
        
    ('-32k_cat', 'llama3_1_8b', 'llama+', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
    ('-slm-32.0.8192.512-32k_cat', 'llama3_1_8b', 'llama+', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
    ('-reattn-32.4x32.8192.512-hu-clip256-32k_cat', 'llama3_1_8b', 'llama+', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 

    ### llama3_2_3b

    ('-32k_cat', 'llama3_2_3b', 'llama+', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
    ('-slm-32.0.8192.512-32k_cat', 'llama3_2_3b', 'llama+', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
    ('-reattn-32.4x32.8192.512-hu-clip256-32k_cat', 'llama3_2_3b', 'llama+', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 

    ### mistral3_7b

    ('-32k_cat', 'mistral3_7b', 'mistral', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
    ('-slm-32.0.8192.512-32k_cat', 'mistral3_7b', 'mistral', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
    ('-reattn-32.4x32.8192.512-hu-clip256-32k_cat', 'mistral3_7b', 'mistral', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 

    ### internlm2_5_7b

    ('-32k_cat', 'internlm2_5_7b', 'internlm2', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
    ('-slm-32.0.8192.512-32k_cat', 'internlm2_5_7b', 'internlm2', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
    ('-reattn-32.4x32.8192.512-hu-clip256-32k_cat', 'internlm2_5_7b', 'internlm2', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 

    ### qwen2_5_7b

    ('-32k_cat', 'qwen2_5_7b', 'qwen2', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
    ('-slm-32.0.8192.512-32k_cat', 'qwen2_5_7b', 'qwen2', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
    ('-reattn-32.4x32.8192.512-hu-clip256-32k_cat', 'qwen2_5_7b', 'qwen2', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
    
    ### qwen2_5_1b

    ('-n1-32k_cat', 'qwen2_5_1b', 'qwen2', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
    ('-slm-32.0.8192.512-32k_cat', 'qwen2_5_1b', 'qwen2', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
    ('-reattn-32.4x32.8192.512-hu-clip256-32k_cat', 'qwen2_5_1b', 'qwen2', '{prompt}', 31500, 
        {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
        'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 

    ## ablation study: 

    # ### chunk_size

    # ('-reattn-32.4x32.4096.512-gu-clip127-32k_cat', 'llama3_8b', 'llama', '{prompt}', 31500,  
    #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
    #     'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
    #     'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
    #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, }),
    # ('-reattn-32.4x32.4096.1024-gu-clip127-qk-32k_cat', 'llama3_8B', 'llama', '{prompt}', 31500,  
    #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
    #     'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
    #     'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
    #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, }),
    # ('-reattn-32.4x32.4096.2048-gu-clip127-qk-32k_cat', 'llama3_8B', 'llama', '{prompt}', 31500,  
    #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 2048, 
    #     'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
    #     'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
    #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, }),

    # ### span size

    # ('-reattn-8.4x8.4096.512-gu-clip511-qk-32k_cat', 'llama3_8B', 'llama', '{prompt}', 31500,  
    #     {'global_size': 64, 'mid_size': 4, 'span_size': 64, 'local_size': 4096, 'chunk_size': 512, 
    #     'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
    #     'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
    #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 511, }),
    # ('-reattn-16.4x16.4096.512-gu-clip255-qk-32k_cat', 'llama3_8B', 'llama', '{prompt}', 31500,  
    #     {'global_size': 64, 'mid_size': 4, 'span_size': 64, 'local_size': 4096, 'chunk_size': 512, 
    #     'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
    #     'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
    #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 255, }),
    # ('-reattn-64.4x64.4096.512-gu-clip63-qk-32k_cat', 'llama3_8B', 'llama', '{prompt}', 31500,  
    #     {'global_size': 64, 'mid_size': 4, 'span_size': 64, 'local_size': 4096, 'chunk_size': 512, 
    #     'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
    #     'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
    #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 63, }),
    # ('-reattn-128.4x128.4096.512-gu-clip31-qk-32k_cat', 'llama3_8B', 'llama', '{prompt}', 31500,  
    #     {'global_size': 32, 'mid_size': 4, 'span_size': 128, 'local_size': 4096, 'chunk_size': 512, 
    #     'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
    #     'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
    #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 31, }),

    # ### top-k

    # ('-reattn-32.1x32.4096.512-gu-clip127-qk-32k_cat', 'llama3_8B', 'llama', '{prompt}', 31500,  
    #     {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
    #     'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
    #     'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
    #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, }),
    # ('-reattn-32.8x32.4096.512-gu-clip127-qk-32k_cat', 'llama3_8B', 'llama', '{prompt}', 31500,  
    #     {'global_size': 32, 'mid_size': 8, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
    #     'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
    #     'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
    #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, }),

    # ### local size

    # ('-reattn-32.4x32.2048.512-gu-clip191-qk-32k_cat', 'llama3_8B', 'llama', '{prompt}', 31500,  
    #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
    #     'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
    #     'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
    #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 191, }),
    # ('-reattn-32.4x32.1024.512-gu-clip224-qk-32k_cat', 'llama3_8B', 'llama', '{prompt}', 31500,  
    #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
    #     'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
    #     'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
    #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 224, }),

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
            max_out_len=500,
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

work_dir = './outputs/reattn_long/'

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

# python run.py eval_xrliu/eval_reattn_long.py --dump-eval-details -r --debug
# python run.py eval_xrliu/eval_reattn_long.py --dump-eval-details -r
