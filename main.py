import os

import aiohttp
import uvicorn
from fastapi import FastAPI
from fastapi import Request
from contextlib import asynccontextmanager
import sys

from ai.llm import Agent, ai_response
from scripts.search import search_in_rag

import logging

from schemas import TextInput

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Запуск приложения: инициализация aiohttp сессии и LLM агента...')
    llm_time_limit = os.getenv("LLM_TIMELIMIT", 60)
    llm_time_limit = aiohttp.ClientTimeout(total=int(llm_time_limit))
    async with aiohttp.ClientSession(timeout=llm_time_limit) as session:
        app.state.session = session
        app.state.agent = Agent(session=session)
        yield
    logger.info("Завершение работы приложения: закрытие сессии...")

app = FastAPI(
    title="RAG Chat Bot",
    description="API для общения с корпоративной базой знаний через RAG",
    version="1.0",
    lifespan=lifespan,
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/chat")
async def chat(request: Request, body_request: TextInput):
    agent = request.app.state.agent
    user_question = body_request.question
    rag_data = search_in_rag(user_question)
    llm_answer = await agent.process(user_question=user_question, rag_response=rag_data)
    logger.info(f'user_question: {user_question}, rag_response: {rag_data}, llm_answer: {llm_answer}')
    return {"message": llm_answer}



if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
