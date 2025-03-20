import os

from datasets import load_from_disk

from opencompass.utils import get_logger

def load_local_dataset(**kwargs):
    path = kwargs.pop("path")
    name = kwargs.pop("name", '')
    split = kwargs.get("split", "")
    data_files = kwargs.get("data_files", None)
    if data_files is not None:
        raise NotImplementedError
    local_path = os.path.join("./data", path, name, split)
    get_logger().info(f"Load hf dataset from local path {local_path} with kwargs {kwargs} ...")
    assert os.path.exists(local_path)
    return load_from_disk(local_path, **kwargs)