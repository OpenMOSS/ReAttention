from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.datasets import AccEvaluator, TruthfulQADatasetForMC


truthfulqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='label',
    train_split='validation',
    test_split='validation')

truthfulqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0: f"{{question}} {{A}}", 
            1: f"{{question}} {{B}}", 
            2: f"{{question}} {{C}}", 
            3: f"{{question}} {{D}}", 
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

truthfulqa_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

truthfulqa_datasets = [
    dict(
        abbr='truthful_qa_mc',
        type=TruthfulQADatasetForMC,
        path='EleutherAI/truthful_qa_mc',
        reader_cfg=truthfulqa_reader_cfg,
        infer_cfg=truthfulqa_infer_cfg,
        eval_cfg=truthfulqa_eval_cfg)
]
