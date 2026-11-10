import os
import yaml

_DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "knowledge_data", "experiments_prompts.yaml"
)

with open(_DATA_FILE, encoding="utf-8") as _f:
    EXPERIMENT_KNOWLEDGE_MAP = yaml.safe_load(_f)


# -------------------- 特殊实验模块（快照分析）提示词 --------------------
# 与实验知识库分离：快照分析不做知识检索，system 提示词由 snapshot_prompts.yaml 整段提供。
# key = 请求体「实验模块」字段值（如 槽式自由搭建 / 槽式场景漫游DCS）。
_SNAPSHOT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "knowledge_data", "snapshot_prompts.yaml"
)

with open(_SNAPSHOT_FILE, encoding="utf-8") as _f:
    SNAPSHOT_PROMPTS_MAP = yaml.safe_load(_f)


def get_snapshot_knowledge(module: str) -> str:
    prompt = SNAPSHOT_PROMPTS_MAP.get(module)
    if prompt is None:
        supported = "、".join(SNAPSHOT_PROMPTS_MAP.keys())
        raise ValueError(f"未知实验模块: {module}，当前支持: {supported}")
    return prompt


def get_knowledge(experiment_key: str) -> str:
    knowledge = EXPERIMENT_KNOWLEDGE_MAP.get(experiment_key)
    if knowledge is None:
        supported = "、".join(EXPERIMENT_KNOWLEDGE_MAP.keys())
        raise ValueError(f"未知实验简称: {experiment_key}，当前支持: {supported}")
    return knowledge
