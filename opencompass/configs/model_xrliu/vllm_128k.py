from opencompass.models.xrliu.vllm_long import VLLMCausalLM

path_dict = {'llama2_7B': '', 
             'llama2_7B_chat': '', 
             'llama2_13B': '', 
             'llama3_8B': '/cpfs01/shared/alillm2/llm_llama3_hf/Meta-Llama-3-8B/', 
             'llama3_8B_chat': '/cpfs01/shared/alillm2/llm_llama3_hf/Meta-Llama-3-8B-Instruct/', 
             'llama3_1_8B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb/', 
             'llama3_1_8B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/07eb05b21d191a58c577b4a45982fe0c049d0693/', 
             'llama3_1_70B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-70B/snapshots/7740ff69081bd553f4879f71eebcc2d6df2fbcb3/', 
             'llama3_2_3B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/392a143b624368100f77a3eafaa4a2468ba50a72/', 
             'internlm2_1B': '',
             'internlm2_1B_base': '',
             'internlm2_7B': '',
             'internlm2_7B_base': '',  
             'internlm2_7B_chat': '',
             'internlm2_20B': '',
             'internlm2.5_7B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--internlm--internlm2_5-7b/snapshots/7bee5c5d7c4591d7f081f5976610195fdd1f1e35/',  # '/nas/shared/alillm2/zhangshuo/exported_transformers/official_Ampere2.5_7B_Enhance_2.0.0/50000/', 
             'internlm2.5_7B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--internlm--internlm2_5-7b-chat-1m/snapshots/8d1a709a04d71440ef3df6ebbe204672f411c8b6/', 

             'qwen1_5_14B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen1.5-14B/snapshots/dce4b190d34470818e5bec2a92cb8233aaa02ca2/', 
             'qwen2_1B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen2-1.5B/snapshots/8a16abf2848eda07cc5253dec660bf1ce007ad7a/', 
             'qwen2_1B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8/', 
             'qwen2_7B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen2-7B/snapshots/d8412313bc00677839b4e38cdc751d536ccc12ab/', 
             'qwen2_7B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen2-7B-Instruct/snapshots/6ddb532fa75db9a1269de86eaa579818eb39743a/', 
             'qwen2_72B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen2-72B/snapshots/e5cebedf0244946b1ad5eb2d753ca3c1a90f11fb/', 
             'mistral3_7B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--mistralai--Mistral-7B-v0.3/snapshots/b67d6a03ca097c5122fa65904fce0413500bf8c8/', 
             'mistral3_7B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de/', 
             'glm4_9B_chat_1M': '/nas/shared/public/llmeval/model_weights/hf_hub/models--THUDM--glm-4-9b-chat-1m/snapshots/715ddbe91082f976ff6a4ca06d59e5bbff6c3642/', 
            }

num_gpus = {'llama2_7B': 4, 'llama2_7B_chat': 4, 'llama2_13B': 8, 
            'llama3_8B': 4, 'llama3_8B_chat': 4, 'llama3_1_8B': 4, 'llama3_1_8B_chat': 4, 'llama3_1_70B': 8, 
            'qwen1_5_1B': 4, 'qwen1_5_7B': 4, 'qwen1_5_14B': 8, 'qwen1_5_32B': 8, 
            'qwen2_1B': 4, 'qwen2_1B_chat': 4, 'qwen2_7B': 4, 'qwen2_7B_chat': 4, 'qwen2_72B': 8, 
            'internlm2_7B': 4, 'internlm2_7B_chat': 4, 
            'internlm2.5_7B': 4, 'internlm2.5_7B_chat': 4, 
            'internlm2_1B': 4, 'internlm2_20B': 4, 
            'mistral3_7B': 4, 'mistral3_7B_chat': 4, 
            'glm4_9B_chat_1M': 4, 
            }

tags = [
        ('-vllm-32k_cat', 'llama3_1_8B_chat', 31500, ), 
        ('-vllm-64k_cat', 'llama3_1_8B_chat', 63500, ), 
        ('-vllm-128k_cat', 'llama3_1_8B_chat', 127500, ), 

        ('-vllm-32k_cat', 'llama3_2_3B_chat', 31500, ), 
        ('-vllm-64k_cat', 'llama3_2_3B_chat', 63500, ), 
        ('-vllm-128k_cat', 'llama3_2_3B_chat', 127500, ), 

        ('-vllm-32k_cat', 'qwen2_7B_chat', 31500, ), 
        ('-vllm-64k_cat', 'qwen2_7B_chat', 63500, ), 
        ('-vllm-128k_cat', 'qwen2_7B_chat', 127500, ), 

        ('-vllm-32k_cat', 'internlm2.5_7B_chat', 31500, ), 
        ('-vllm-64k_cat', 'internlm2.5_7B_chat', 63500, ), 
        ('-vllm-128k_cat', 'internlm2.5_7B_chat', 127500, ), 
    ]

models = []

for abbr, group, cat_len in tags:
    models.append(
        dict(
            type=VLLMCausalLM,
            abbr=f'{group}{abbr}',
            path=path_dict[group],
            max_out_len=50,
            max_seq_len=1048576, 
            long_bench_cat=cat_len, 
            batch_size=1,
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', 
                                  use_fast=False, trust_remote_code=True), 
            run_cfg=dict(num_gpus=1,  # num_gpus[group.split('-')[0]], 
                         num_procs=1),
            end_str='<eoa>',
        )
    )