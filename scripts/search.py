from dotenv import load_dotenv

load_dotenv()

from rag.ingestion import get_embeddings

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


def main(question):

    client = QdrantClient(
        host="localhost",
        port=6333
    )

    embeddings = get_embeddings()

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="corporate_docs",
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


if __name__ == "__main__":
    user_question = input("Введите свой вопрос: ")
    main(user_question)