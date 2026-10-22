import os
import sys

from rag.ingestion import get_embeddings

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


def search_in_rag(question):

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT
    )

    embeddings = get_embeddings()

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    results = vectorstore.similarity_search(
        query=question,
        k=5
    )

    logger.info(f"Вопрос: {question}")

    for i, doc in enumerate(results, start=1):
        logger.info(f"[{i}] Источник: {doc.metadata.get('source')}")
        logger.info(f"Отдел: {doc.metadata.get('department')}")
        logger.info(doc.page_content[:300])
        logger.info("-" * 80)

    return results

if __name__ == "__main__":
    user_question = input("Введите свой вопрос: ")
    logger.info(search_in_rag(user_question))