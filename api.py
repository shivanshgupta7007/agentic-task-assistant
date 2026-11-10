"""Minimal web API for the agent, so the loop can be tried from a browser.

Run with: uvicorn api:app --host 0.0.0.0 --port 8000
Requires GEMINI_API_KEY in the environment (see README).

POST /run  {"task": "..."}  ->  {"answer": ..., "steps": n, "trace": [...]}
GET  /health               ->  {"status": "ok"}
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.core import Agent
from agent.llm import GeminiAgentModel
from agent.tools import ToolRegistry

app = FastAPI(title="Agentic Task Assistant", version="1.0.0")


class RunRequest(BaseModel):
    task: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "gemini-3.6-flash"}


@app.post("/run")
def run(req: RunRequest) -> dict:
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured on the server.")
    if not req.task.strip():
        raise HTTPException(status_code=422, detail="task must not be empty")
    tools = ToolRegistry()
    agent = Agent(model=GeminiAgentModel(tools), tools=tools, max_steps=8)
    result = agent.run(req.task)
    return {"answer": result.answer, "steps": result.steps, "trace": result.trace}
