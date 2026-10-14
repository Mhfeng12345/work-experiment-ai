import json
import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from knowledge_base import get_knowledge, EXPERIMENT_KNOWLEDGE_MAP
from prompts import EXPERIMENT_PROMPT

load_dotenv()

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "18008"))

app = FastAPI(title="Experiment AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "qwen/qwen3-vl-4b"),
    base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:19874/v1"),
    api_key="not-needed",
    temperature=0.2,
)

chain = EXPERIMENT_PROMPT | llm


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
    experiment_name: str
    tasks: list[TaskData] = Field(default_factory=list)
    questions: list[QuestionData] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    experiment_data: ExperimentData


class ChatRequest(BaseModel):
    message: str = Field(description="用户输入的聊天内容")


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
    """查看当前支持的实验"""
    return {"supported": list(EXPERIMENT_KNOWLEDGE_MAP.keys())}


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    exp_name = request.experiment_data.experiment_name

    try:
        knowledge = get_knowledge(exp_name)
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
        chain.astream(chain_input),
        error_prefix=f"分析失败 [{exp_name}]"
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    """最简聊天接口：输入字符串，流式返回回复（暂无记忆、无知识库）"""
    return await stream_response(
        llm.astream([HumanMessage(content=request.message)]),
        error_prefix="聊天失败"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)