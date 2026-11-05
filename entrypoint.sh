#!/bin/sh
set -e

# 首次启动：数据卷为空时从镜像内置种子拷贝 chroma_db + 知识文档
if [ ! -f /app/vector_db/chroma_db/chroma.sqlite3 ]; then
    echo "[entrypoint] 初始化向量库数据卷…"
    mkdir -p /app/vector_db
    cp -rn /opt/seed/vector_db/* /app/vector_db/ 2>/dev/null || true
    echo "[entrypoint] 初始化完成"
fi

# 单 worker：Chroma 嵌入式 + 每进程一份 embedding 模型，不要加 --workers
exec python -m uvicorn main:app --host "$APP_HOST" --port "$APP_PORT"
