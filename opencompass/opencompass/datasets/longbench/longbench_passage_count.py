from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path
# from opencompass.utils.internal.load_dataset import \
#     load_local_dataset as load_dataset

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class LongBenchpassage_countDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path)
        dataset = load_dataset(path=path,
                               name=name,
                               data_dir=path,
                               trust_remote_code=True)
        # if 'data_files' in kwargs:
        #     kwargs['data_files'] = get_data_path(kwargs['data_files'],
        #                                          local_mode=True)
        # dataset = load_dataset(**kwargs)
        split = 'test'
        raw_data = []
        for i in range(len(dataset[split])):
            context = dataset[split]['context'][i]
            answers = dataset[split]['answers'][i]
            raw_data.append({'context': context, 'answers': answers})
        dataset[split] = Dataset.from_list(raw_data)
        return dataset
