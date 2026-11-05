# experiment-ai 后端镜像（linux/amd64）
# 结构还原自 experiment-ai:0.2.2（docker history），0.2.3 起构建上下文完整入库。
# 构建：docker build -t experiment-ai:0.2.3 .

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/tmp \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf SENTENCE_TRANSFORMERS_HOME=/tmp/st

# 基础运行依赖（libgomp1 供 torch CPU、curl 供 HEALTHCHECK）
RUN sed -i 's|http://deb.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 依赖（torch CPU 用 wheels，其余用 lock 文件，镜像源清华）
COPY requirements.lock.linux.txt requirements.lock.txt
COPY wheels/ /wheels/
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --find-links /wheels torch==2.14.1+cpu
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.lock.txt

# 应用代码与数据（知识外置：knowledge_data 含 experiments_prompts.yaml + docs/*.md）
COPY main.py prompts.py knowledge_base.py chroma_controller.py validate_knowledge.py ./
COPY knowledge_data/ ./knowledge_data/
COPY model/embedding/bge-small-zh-v1.5/ ./model/embedding/bge-small-zh-v1.5/
COPY test_data/ ./test_data/
COPY vector_db/ ./vector_db/
COPY entrypoint.sh /app/entrypoint.sh

# 种子：首次启动空卷时由 entrypoint 从 /opt/seed 拷贝
RUN chmod +x /app/entrypoint.sh && mkdir -p /opt/seed && cp -r vector_db /opt/seed/vector_db

ENV APP_HOST=0.0.0.0 APP_PORT=18741 LLM_MODEL=qwen3.5-4b \
    LLM_BASE_URL=http://llama-server:8080/v1 LLM_MAX_TOKENS=2048 \
    AI_MODEL_NAME=qwen3.5-4b AI_MODEL_VERSION=4B-Q4_K_M \
    AI_MODEL_DEPLOYMENT=私有化部署 AI_MODEL_PROVIDER=虚拟仿真实验平台 \
    APP_VERSION=0.2.3

EXPOSE 18741
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:18741/health || exit 1
ENTRYPOINT ["/app/entrypoint.sh"]
