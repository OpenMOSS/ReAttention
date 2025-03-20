from mmengine.config import read_base

with read_base():

    # shot task for long_score

    # from ..datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # from ..datasets.mbpp.mbpp_gen_830460 import mbpp_datasets

    # from ..datasets.bbh.bbh_gen_5b92b0 import bbh_datasets

    # from ..datasets.math.math_gen_265cce import math_datasets  # don't use
    from ..datasets.gsm8k.gsm8k_gen_17d0dc import gsm8k_datasets  # modified in 240907

    # from ..datasets.ARC_e.ARC_e_ppl_a450bd import ARC_e_datasets                             # zero shot
    # from ..datasets.ARC_c.ARC_c_ppl_a450bd import ARC_c_datasets                             # zero shot

    # from ..datasets.hellaswag.hellaswag_ppl_47bff9 import hellaswag_datasets                 # zero shot

    # from ..datasets.winogrande.winogrande_ppl_55a66e import winogrande_datasets              # zero shot
    # from ..datasets.truthfulqa.truthfulqa_gen_mc import truthfulqa_datasets                  # zero shot
    
    # from ..datasets.SuperGLUE_AX_b.SuperGLUE_AX_b_ppl_6db806 import AX_b_datasets            # zero shot
    # from ..datasets.SuperGLUE_AX_g.SuperGLUE_AX_g_ppl_66caf3 import AX_g_datasets            # zero shot
    # from ..datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl_314b96 import BoolQ_datasets         # zero shot
    # from ..datasets.SuperGLUE_CB.SuperGLUE_CB_ppl_0143fe import CB_datasets                  # zero shot
    # from ..datasets.SuperGLUE_COPA.SuperGLUE_COPA_ppl_9f3618 import COPA_datasets            # zero shot
    # from ..datasets.SuperGLUE_MultiRC.SuperGLUE_MultiRC_ppl_ced824 import MultiRC_datasets   # zero shot
    # from ..datasets.SuperGLUE_RTE.SuperGLUE_RTE_ppl_66caf3 import RTE_datasets               # zero shot
    # from ..datasets.SuperGLUE_ReCoRD.SuperGLUE_ReCoRD_gen_30dea0 import ReCoRD_datasets      # zero shot
    # from ..datasets.SuperGLUE_WiC.SuperGLUE_WiC_ppl_312de9 import WiC_datasets               # zero shot
    # from ..datasets.SuperGLUE_WSC.SuperGLUE_WSC_ppl_003529 import WSC_datasets               # zero shot

    from ..datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets                                # 5-shot
    # from ..datasets.ceval.ceval_ppl_578f8d import ceval_datasets                             # 5-shot
    from ..datasets.cmmlu.cmmlu_ppl_8b9c76 import cmmlu_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
