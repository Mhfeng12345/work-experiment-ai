import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from knowledge_base import CURRENT_EXP_KNOWLEDGE
from prompts import EXPERIMENT_PROMPT

# 必须在 ChatOpenAI 初始化之前加载，否则 os.getenv 只能拿到代码里的默认值。
load_dotenv()


app = FastAPI(title="Experiment AI API")

# WebGL 浏览器跨域测试配置。
# 正式部署时，建议将 * 替换为实际前端域名。
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
    max_tokens=2048,
)

chain = EXPERIMENT_PROMPT | llm


class AnalyzeRequest(BaseModel):
    experiment_data: dict[str, Any] = Field(
        description="Unity 传入的实验数据 JSON"
    )


class AnalyzeResponse(BaseModel):
    report: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    try:
        experiment_json = json.dumps(
            request.experiment_data,
            ensure_ascii=False,
            indent=2,
        )

        result = await chain.ainvoke({
            "current_exp_knowledge": CURRENT_EXP_KNOWLEDGE,
            "experiment_data": experiment_json,
        })

        return AnalyzeResponse(report=str(result.content))

    except Exception as e:
        # MVP 调试阶段便于查看问题；正式部署不要向客户端暴露详细内部异常。
        raise HTTPException(
            status_code=500,
            detail=f"实验分析失败：{str(e)}",
        )