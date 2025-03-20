from .accessory import LLaMA2AccessoryModel  # noqa: F401
from .ai360_api import AI360GPT  # noqa: F401
from .alaya import AlayaLM  # noqa: F401
from .baichuan_api import BaiChuan  # noqa: F401
from .baidu_api import ERNIEBot  # noqa: F401
from .bailing_api_oc import BailingAPI  # noqa: F401
from .base import BaseModel, LMTemplateParser  # noqa: F401
from .base_api import APITemplateParser, BaseAPIModel  # noqa: F401
from .bytedance_api import ByteDance  # noqa: F401
from .claude_allesapin import ClaudeAllesAPIN  # noqa: F401
from .claude_api import Claude  # noqa: F401
from .claude_sdk_api import ClaudeSDK  # noqa: F401
from .deepseek_api import DeepseekAPI  # noqa: F401
from .doubao_api import Doubao  # noqa: F401
from .gemini_api import Gemini  # noqa: F401
from .glm import GLM130B  # noqa: F401
from .huggingface import HuggingFace  # noqa: F401
from .huggingface import HuggingFaceCausalLM  # noqa: F401
from .huggingface import HuggingFaceChatGLM3  # noqa: F401
from .huggingface_above_v4_33 import HuggingFaceBaseModel  # noqa: F401
from .huggingface_above_v4_33 import HuggingFacewithChatTemplate  # noqa: F401
from .hunyuan_api import Hunyuan  # noqa: F401
from .intern_model import InternLM  # noqa: F401
from .interntrain import InternTrain  # noqa: F401
from .krgpt_api import KrGPT  # noqa: F401
from .lightllm_api import LightllmAPI, LightllmChatAPI  # noqa: F401
from .llama2 import Llama2, Llama2Chat  # noqa: F401
from .minimax_api import MiniMax, MiniMaxChatCompletionV2  # noqa: F401
from .mistral_api import Mistral  # noqa: F401
from .mixtral import Mixtral  # noqa: F401
from .modelscope import ModelScope, ModelScopeCausalLM  # noqa: F401
from .moonshot_api import MoonShot  # noqa: F401
from .nanbeige_api import Nanbeige  # noqa: F401
from .openai_api import OpenAI  # noqa: F401
from .openai_api import OpenAISDK  # noqa: F401
from .pangu_api import PanGu  # noqa: F401
from .qwen_api import Qwen  # noqa: F401
from .rendu_api import Rendu  # noqa: F401
from .sensetime_api import SenseTime  # noqa: F401
from .stepfun_api import StepFun  # noqa: F401
from .turbomind import TurboMindModel  # noqa: F401
from .turbomind_with_tf_above_v4_33 import \
    TurboMindModelwithChatTemplate  # noqa: F401
from .unigpt_api import UniGPT  # noqa: F401
from .vllm import VLLM  # noqa: F401
from .vllm_with_tf_above_v4_33 import VLLMwithChatTemplate  # noqa: F401
from .xunfei_api import XunFei, XunFeiSpark  # noqa: F401
from .yayi_api import Yayi  # noqa: F401
from .yi_api import YiAPI  # noqa: F401
from .zhipuai_api import ZhiPuAI  # noqa: F401
from .zhipuai_v2_api import ZhiPuV2AI  # noqa: F401

# note: 
# 每个人自己提交的时候把自己的类都注释掉再提交，提交完后再解除注释使用
# 使用过程中，如果使用别人已经实现的类，先复制到自己的路径下再修改使用

from .xrliu.re_attention.re_attention_wrapper import LightCacheCausalLM
from .xrliu.huggingface_long import HuggingFaceModel
from .xrliu.infllm_model import InfLLM_xrliu
from .xrliu.vllm_long import VLLMCausalLM

# from .zgliu.llama2_h2o import Llama2_H2O
# from .zgliu.llama2_snapkv import Llama2_SnapKV
# from .zgliu.llama2_streamingllm import Llama2_StreamingLLM
# from .zgliu.infllm_model import INFLLM_LlamaForCausalLM
# from .zgliu.kivi_model import KIVI_LlamaForCausalLM
# from .zgliu.lightcache_model import LightCache_MossHuaweiForCausalLM
# from .zgliu.rag_context_model import RAG_CONTEXT_LlamaForCausalLM
# from .zgliu.selfextend_model import SelfExtend_LlamaForCausalLM
# from .zgliu.llama2_chunkllama import ChunkLlama_LlamaForCausalLM
# from .zgliu.llama2_rerope import Rerope_LlamaForCausalLM