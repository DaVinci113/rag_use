import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from dotenv import load_dotenv

EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL")


def load_document(file_path: str):
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    elif suffix == ".docx":
        loader = Docx2txtLoader(file_path)
    elif suffix == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    elif suffix == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Неподдерживаемый формат: {suffix}")

    return loader.load()


def load_all_documents(root_dir: str):
    documents = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)

            try:
                docs = load_document(file_path)
                documents.extend(docs)
            except Exception as e:
                print(f"Ошибка при загрузке {file_path}: {e}")

    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    return chunks

def enrich_metadata(chunks):
    enriched_chunks = []

    for chunk in chunks:
        source = chunk.metadata.get("source", "")

        if "hr/" in source:
            department = "hr"
        elif "it/" in source:
            department = "it"
        elif "legal/" in source:
            department = "legal"
        else:
            department = "general"

        chunk.metadata["department"] = department
        chunk.metadata["access_level"] = "internal"
        chunk.metadata["chunk_text_preview"] = chunk.page_content[:200]

        enriched_chunks.append(chunk)

    return enriched_chunks

def create_qdrant_collection(collection_name: str):
    client = QdrantClient(host="localhost", port=6333)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE,
            ),
        )

    return client


def index_documents(chunks):
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    client = QdrantClient(host="localhost", port=6333)

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="corporate_docs_v2",
        embedding=embeddings,
    )

    vectorstore.add_documents(chunks)

def delete_collection(collection_name: str):
    client = QdrantClient(
        host="localhost",
        port=6333
    )

    client.delete_collection(collection_name)

if __name__ == '__main__':
    delete_collection("corporate_docs")
