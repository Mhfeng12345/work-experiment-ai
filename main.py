import asyncio
import json
import os
import threading
import time
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from knowledge_base import get_knowledge, EXPERIMENT_KNOWLEDGE_MAP
from prompts import EXPERIMENT_PROMPT, QA_PROMPT

from vector_db.knowledge_store import (
    list_experiment_keys,
    search_knowledge,
    warmup,
)

load_dotenv()

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "18741"))

app = FastAPI(title="Experiment AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "qwen/qwen3.5-9b"),
    base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:19874/v1"),
    api_key="not-needed",
    temperature=0.2,
    max_tokens=2048,
)

chain = EXPERIMENT_PROMPT | llm
qa_chain = QA_PROMPT | llm


# -------------------- 启动预热 --------------------

def _warmup_worker() -> None:
    t0 = time.perf_counter()
    try:
        count = warmup()
        print(f"[warmup] 知识库预热完成：{count} 个知识点，耗时 {time.perf_counter() - t0:.2f}s")
    except Exception as e:
        print(f"[warmup] 预热失败：{e}")


threading.Thread(target=_warmup_worker, daemon=True).start()


# -------------------- 数据模型 --------------------
class StepData(BaseModel):
    name: str
    status: str


class TaskData(BaseModel):
    name: str
    steps: list[StepData] = Field(default_factory=list)


class QuestionData(BaseModel):
    question: str
    option: list[str] = Field(default_factory=list)
    answer: str | None = ""
    correct_answer: str | None = ""


class ExperimentData(BaseModel):
    experiment_key: str
    tasks: list[TaskData] = Field(default_factory=list)
    questions: list[QuestionData] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    experiment_data: ExperimentData


class ChatRequest(BaseModel):
    message: str = Field(description="用户输入的聊天内容")


class ExperimentChatRequest(BaseModel):
    experiment_key: str = Field(description="实验简称")
    question: str = Field(min_length=1, description="用户的问题")


# -------------------- 公共流式输出 --------------------
async def stream_response(generator, error_prefix: str = "请求失败"):
    """统一流式输出封装"""

    async def event_stream():
        try:
            async for chunk in generator:
                if not chunk.content:
                    continue
                delta = {"choices": [{"delta": {"content": chunk.content}}]}
                yield f"data: {json.dumps(delta, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"{error_prefix}: {e}")
            delta = {"choices": [{"delta": {"content": f"\n\n> {error_prefix}：{e}"}}]}
            yield f"data: {json.dumps(delta, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# -------------------- 接口 --------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/experiments")
async def list_experiments():
    """查看当前支持的实验简称"""
    return {"supported": await asyncio.to_thread(list_experiment_keys)}


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    exp_key = request.experiment_data.experiment_key

    try:
        knowledge = get_knowledge(exp_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    experiment_json = json.dumps(
        request.experiment_data.model_dump(),
        ensure_ascii=False,
        indent=2,
    )

    chain_input = {
        "current_exp_knowledge": knowledge,
        "experiment_data": experiment_json,
    }

    return await stream_response(
        chain.astream(chain_input, stream_mode="messages"),
        error_prefix=f"分析失败 [{exp_key}]",
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    """最简聊天接口：输入字符串，流式返回回复（暂无记忆、无知识库）"""
    return await stream_response(
        llm.astream([HumanMessage(content=request.message)]), error_prefix="聊天失败"
    )


@app.post("/knowledgeDBChat")
async def experiment_chat(request: ExperimentChatRequest):
    """基于指定实验知识库的 RAG 问答"""
    exp_key = request.experiment_key

    try:
        docs = await asyncio.to_thread(
            search_knowledge, request.question, experiment_key=exp_key, k=4
        )
    except ValueError as e:
        supported = "、".join(await asyncio.to_thread(list_experiment_keys))
        raise HTTPException(
            status_code=400,
            detail=f"{e}，当前支持: {supported}",
        )

    if not docs:
        raise HTTPException(
            status_code=404,
            detail=f"未检索到实验「{exp_key}」的相关知识",
        )

    context = "\n\n".join(doc.page_content for doc in docs)

    return await stream_response(
        qa_chain.astream(
            {"context": context, "question": request.question},
            stream_mode="messages",
        ),
        error_prefix=f"知识问答失败 [{exp_key}]",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
