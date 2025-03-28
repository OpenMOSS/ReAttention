# Most of the code in this file is copied from https://github.com/openai/simple-evals/blob/main/math_eval.py
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.summarizers import DefaultSubjectiveSummarizer


with read_base():
    from opencompass.configs.datasets.SimpleQA.simpleqa_gen import simpleqa_datasets
    from opencompass.configs.models.openai.gpt_4o_2024_05_13 import models as gpt_4o_2024_05_13_model

models = gpt_4o_2024_05_13_model # model for generation
judge_models = gpt_4o_2024_05_13_model # model for evaluation

datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])
summarizer = dict(type=DefaultSubjectiveSummarizer)

# -------------Inferen Stage ----------------------------------------

from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask)
    ),
)

eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        models=[gpt_4o_2024_05_13_model],
        judge_models=[gpt_4o_2024_05_13_model],
    ),
    runner=dict(type=LocalRunner,
                max_num_workers=256,
                task=dict(type=SubjectiveEvalTask)),
)
