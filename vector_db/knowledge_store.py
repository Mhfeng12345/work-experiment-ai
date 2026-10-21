
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# -------------------- 路径与配置 --------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge_docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
EMBEDDING_MODEL_DIR = PROJECT_DIR / "model" / "embedding" / "bge-small-zh-v1.5"

COLLECTION_NAME = "experiment_knowledge"

_db = None


# -------------------- 初始化向量数据库 --------------------

def get_db() -> Chroma:
    global _db

    if _db is None:
        if not EMBEDDING_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Embedding 模型不存在：{EMBEDDING_MODEL_DIR}"
            )

        embeddings = HuggingFaceEmbeddings(
            model_name=str(EMBEDDING_MODEL_DIR),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        _db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    return _db


# -------------------- 解析 Markdown --------------------

def load_markdown(file_path: Path) -> list[Document]:
    text = file_path.read_text(encoding="utf-8")

    metadata = {
        "experiment_key": "",
        "experiment_name": "",
        "source": file_path.name,
    }

    # 读取文件开头的 metadata
    if text.startswith("---"):
        parts = text.split("---", 2)

        if len(parts) == 3:
            for line in parts[1].splitlines():
                key, separator, value = line.partition(":")

                if separator and key.strip() in metadata:
                    metadata[key.strip()] = value.strip().strip("'\"")

            text = parts[2]

    if not metadata["experiment_key"]:
        raise ValueError(f"缺少 experiment_key：{file_path.name}")

    if not metadata["experiment_name"]:
        raise ValueError(f"缺少 experiment_name：{file_path.name}")

    documents = []
    chapter = ""
    point_title = ""
    content_lines = []

    def save_point():
        if not point_title:
            return

        content = "\n".join(content_lines).strip()

        if not content:
            return

        # 章节标题、知识点标题都参与 Embedding
        page_content = f"{chapter}\n{point_title}\n{content}".strip()

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata.copy(),
            )
        )

    for line in text.splitlines():
        heading = re.match(r"^(#{1,2})\s+(.+)$", line)

        if heading and len(heading.group(1)) == 1:
            save_point()
            chapter = heading.group(2).strip()
            point_title = ""
            content_lines = []

        elif heading and len(heading.group(1)) == 2:
            save_point()
            point_title = heading.group(2).strip()
            content_lines = []

        elif point_title:
            content_lines.append(line)

    save_point()

    if not documents:
        raise ValueError(f"没有解析到有效知识点：{file_path.name}")

    return documents


# -------------------- 导入与更新知识 --------------------

def import_markdown(file_path: Path) -> int:
    db = get_db()
    documents = load_markdown(file_path)
    source = documents[0].metadata["source"]

    # 删除同一来源的旧数据，避免更新后残留旧知识
    existing = db.get(where={"source": source})
    old_ids = existing.get("ids", [])

    if old_ids:
        db.delete(ids=old_ids)

    # 每个知识点生成独立 ID
    ids = [
        f"{source}::{index:04d}"
        for index in range(1, len(documents) + 1)
    ]

    db.add_documents(documents=documents, ids=ids)

    return len(documents)


def import_all() -> None:
    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(f"知识文档目录不存在：{KNOWLEDGE_DIR}")

    files = sorted(KNOWLEDGE_DIR.glob("*.md"))

    if not files:
        print(f"未找到 Markdown 文件：{KNOWLEDGE_DIR}")
        return

    total = 0

    for file_path in files:
        count = import_markdown(file_path)
        total += count
        print(f"导入成功：{file_path.name}，{count} 个知识点")

    print(f"\n导入完成，共处理 {len(files)} 个文件、{total} 个知识点。")


def warmup() -> int:
    """加载 embedding 模型与向量库，并跑一次真实编码路径（触发 tokenizer 等懒加载）"""
    db = get_db()
    db.similarity_search("预热", k=1)
    return db._collection.count()


# -------------------- 实验简称匹配 --------------------

def list_experiment_keys() -> list[str]:
    """读取向量库中实际存在的全部实验简称"""
    metadatas = get_db()._collection.get(include=["metadatas"])["metadatas"]

    keys = {
        (m or {}).get("experiment_key")
        for m in metadatas
        if (m or {}).get("experiment_key")
    }

    return sorted(keys)


def resolve_experiment_key(experiment_key: str) -> str | None:
    """返回库中真实存在的实验简称；未命中返回 None"""
    key = experiment_key.strip()
    return key if key in list_experiment_keys() else None


# -------------------- 知识库维护（查 / 删） --------------------

def count_knowledge() -> int:
    """返回向量库中的知识点总数"""
    return get_db()._collection.count()


def list_sources() -> list[str]:
    """返回向量库中所有来源文件名（去重、排序）"""
    metadatas = get_db().get(include=["metadatas"]).get("metadatas") or []

    return sorted(
        {
            (meta or {}).get("source")
            for meta in metadatas
            if (meta or {}).get("source")
        }
    )


def list_documents(
    experiment_key: str | None = None,
    source: str | None = None,
    limit: int | None = None,
) -> list[Document]:
    """按实验简称 / 来源筛选知识点；都不传则返回全部"""
    where: dict = {}

    if experiment_key:
        where["experiment_key"] = experiment_key

    if source:
        where["source"] = source

    result = get_db().get(
        where=where or None,
        limit=limit,
        include=["documents", "metadatas"],
    )

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(documents, metadatas)
    ]


def get_document(doc_id: str) -> Document | None:
    """按 id 获取单条知识点，不存在返回 None"""
    found = get_db().get_by_ids([doc_id])

    return found[0] if found else None


def delete_source(source: str) -> int:
    """删除指定来源（文件名）的全部知识点，返回删除条数"""
    db = get_db()
    ids = db.get(where={"source": source}).get("ids") or []

    if ids:
        db.delete(ids=ids)

    return len(ids)


def delete_experiment(experiment_key: str) -> int:
    """删除指定实验的全部知识点，返回删除条数"""
    resolved = resolve_experiment_key(experiment_key)

    if resolved is None:
        raise ValueError(f"未知实验简称: {experiment_key}")

    db = get_db()
    ids = db.get(where={"experiment_key": resolved}).get("ids") or []

    if ids:
        db.delete(ids=ids)

    return len(ids)


def delete_document(doc_id: str) -> bool:
    """按 id 删除单条知识点，返回是否删除了记录"""
    db = get_db()

    if not db.get_by_ids([doc_id]):
        return False

    db.delete(ids=[doc_id])
    return True


# -------------------- 语义检索 --------------------

def search_knowledge(
    question: str,
    experiment_key: str | None = None,
    k: int = 3,
) -> list[Document]:
    if k < 1:
        raise ValueError("k 必须大于等于 1")

    if experiment_key:
        resolved = resolve_experiment_key(experiment_key)

        if resolved is None:
            raise ValueError(f"未知实验简称: {experiment_key}")

        search_filter = {"experiment_key": resolved}
    else:
        search_filter = None

    return get_db().similarity_search(
        query=question,
        k=k,
        filter=search_filter,
    )
