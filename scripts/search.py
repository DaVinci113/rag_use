import logging
import os
import sys

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore

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


def search_in_rag(question, qdrant_client, embeddings):

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    results = vectorstore.similarity_search_with_score(query=question, k=5)
    filtered = [(doc, score) for doc, score in results if score > 0.3]

    logger.info(f'filtered raw rag data: {len(filtered)}')

    return filtered

if __name__ == "__main__":
    user_question = input("Введите свой вопрос: ")
