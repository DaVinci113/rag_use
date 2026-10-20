import os

from rag.ingestion import get_embeddings

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


def main(question):

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

    print(f"Вопрос: {question}")
    print()

    for i, doc in enumerate(results, start=1):
        print(f"[{i}] Источник: {doc.metadata.get('source')}")
        print(f"Отдел: {doc.metadata.get('department')}")
        print(doc.page_content[:300])
        print("-" * 80)

    return results

if __name__ == "__main__":
    user_question = input("Введите свой вопрос: ")
    print(main(user_question))