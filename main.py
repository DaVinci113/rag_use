import logging
import os
import sys
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from qdrant_client import QdrantClient

from ai.llm import Agent
from rag.ingestion import get_embeddings
from schemas import TextInput
from scripts.search import search_in_rag

load_dotenv()

qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))

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
    logger.info(f'Подключение к Qdrant: {qdrant_host}:{qdrant_port}')
    app.state.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
    logger.info('Загружаем модель эмбеддингов...')
    app.state.embeddings = get_embeddings()
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
    qdrant_client = request.app.state.qdrant_client
    embeddings = request.app.state.embeddings
    user_question = body_request.question
    try:
        logger.info(f'Сбор RAG данных по запросу: {user_question}')
        rag_data = search_in_rag(user_question, qdrant_client, embeddings)
        rag_context = []
        for doc, score in rag_data:
            logger.info(f'doc: {doc}, score: {score}\n\n')
            rag_context.append(f"\n\n[Источник: {doc.metadata.get('source')}]\n{doc.page_content}")
        final_rag = ''.join(rag_context)
        logger.info(f'Отправка в LLM user_question: {user_question} rag_context: {final_rag}')
        llm_answer = await agent.process(user_question=user_question, rag_response=final_rag)
        logger.info(f'user_question: {user_question}, rag_response: {rag_data}, llm_answer: {llm_answer}')
        return {"message": llm_answer}
    except Exception as e:
        logger.error(e)
        return {"message": str(e)}



if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
