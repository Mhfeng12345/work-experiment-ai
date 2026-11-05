import os
import yaml

_DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "knowledge_data", "experiments_prompts.yaml"
)

with open(_DATA_FILE, encoding="utf-8") as _f:
    EXPERIMENT_KNOWLEDGE_MAP = yaml.safe_load(_f)


def get_knowledge(experiment_key: str) -> str:
    knowledge = EXPERIMENT_KNOWLEDGE_MAP.get(experiment_key)
    if knowledge is None:
        supported = "、".join(EXPERIMENT_KNOWLEDGE_MAP.keys())
        raise ValueError(f"未知实验简称: {experiment_key}，当前支持: {supported}")
    return knowledge
