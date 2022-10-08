from dataclasses import dataclass
from typing import Dict, List
import yaml
import json

from logging import getLogger
LOG = getLogger(__name__)

@dataclass
class Config:
    db_path: str = "./data/db.sqlite"
    dataset_path: str = "./data/dataset.json"
    index_path: str = "./data/search.index"

def read_dataset(dataset_path: str) -> List[Dict[str, str]]:
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    return dataset

def get_config(config_path: str) -> Config:
    with open(config_path, "r") as stream:
        try:
            config = Config(**yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            LOG.error(exc)
            config = Config()

    return config


