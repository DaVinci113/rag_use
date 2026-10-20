import os
from pathlib import Path

# Document — базовый класс LangChain для документа.
# Импортируется из langchain_core (стабильное ядро, НЕ deprecated).
# Содержит два поля: page_content (текст) и metadata (словарь метаданных)
from langchain_core.documents import Document

# PdfReader — чтение PDF напрямую через pypdf.
# Легче и быстрее, чем PyPDFLoader из langchain-community
from pypdf import PdfReader

# docx2txt — извлечение текста из .docx файлов
import docx2txt

# Сплиттер для чанкинга — из актуального отдельного пакета
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Локальные эмбеддинги через HuggingFace (как договаривались ранее)
from langchain_huggingface import HuggingFaceEmbeddings

# Интеграция LangChain + Qdrant — отдельный актуальный пакет
from langchain_qdrant import QdrantVectorStore

# Клиент и модели Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Загрузка переменных окружения
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "unsloth/bge-small-en-v1.5")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "corporate_docs")
QDRANT_PORT = os.getenv("QDRANT_PORT", 6333)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")


def load_document(file_path: str) -> list[Document]:
    """
    Загружает один документ и возвращает список объектов Document.
    Парсинг делаем напрямую, без langchain-community.
    """

    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        # Читаем PDF постранично, чтобы сохранить номера страниц в метаданных
        reader = PdfReader(str(path))

        documents = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            # Пропускаем пустые страницы (сканы без текстового слоя)
            if not text.strip():
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "page": page_num + 1,  # человеческая нумерация с 1
                    },
                )
            )

        return documents

    elif suffix == ".docx":
        # docx2txt возвращает весь текст одним куском
        text = docx2txt.process(str(path))

        return [
            Document(
                page_content=text,
                metadata={"source": str(path)},
            )
        ]

    elif suffix in (".txt", ".md"):
        # Текстовые файлы читаем напрямую.
        # encoding="utf-8" обязателен для русского текста
        text = path.read_text(encoding="utf-8")

        return [
            Document(
                page_content=text,
                metadata={"source": str(path)},
            )
        ]

    else:
        # Вместо исключения просто пропускаем неподдерживаемый файл —
        # так один "мусорный" файл не уронит всю индексацию
        print(f"Пропускаем неподдерживаемый формат: {file_path}")
        return []


def load_all_documents(root_dir: str) -> list[Document]:
    print("Загрузка всех документов...")
    """
    Рекурсивно обходит папку и загружает все поддерживаемые документы.
    """
    documents = []
    try:
        for root, _, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)

                try:
                    docs = load_document(file_path)
                    documents.extend(docs)
                except Exception as e:
                    # Логируем ошибку и продолжаем работу со следующими файлами
                    print(f"Ошибка при загрузке {file_path}: {e}")
    except Exception as e:
        print(e)
    print("Завершение загрузки всех документов")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Разбивает документы на чанки.
    split_documents() автоматически копирует metadata из исходных
    документов в каждый чанк.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",  # абзацы
            "\n",    # строки
            ". ",    # предложения
            " ",     # слова
            "",      # символы
        ],
    )

    return splitter.split_documents(documents)


def enrich_metadata(chunks: list[Document]) -> list[Document]:
    """
    Добавляет бизнес-метаданные к чанкам.
    """
    enriched_chunks = []

    for chunk in chunks:
        source = chunk.metadata.get("source", "")

        # Определяем отдел по пути к файлу
        if "hr/" in source or "hr\\" in source:   # "\\" — для Windows-путей
            department = "hr"
        elif "it/" in source or "it\\" in source:
            department = "it"
        elif "legal/" in source or "legal\\" in source:
            department = "legal"
        else:
            department = "general"

        chunk.metadata["department"] = department
        chunk.metadata["access_level"] = "internal"
        chunk.metadata["chunk_text_preview"] = chunk.page_content[:200]

        enriched_chunks.append(chunk)

    return enriched_chunks


def get_embeddings():
    """
    Локальные эмбеддинги через HuggingFace.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # "cuda" если есть GPU
        encode_kwargs={"normalize_embeddings": True},
    )


def index_documents(chunks: list[Document], collection_name: str = COLLECTION_NAME):
    """
    Индексирует чанки в Qdrant.
    """
    if not chunks:
        print("Чанков нет. Индексация пропущена.")
        return None

    embeddings = get_embeddings()
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Определяем размерность динамически — не зашиваем число в код.
    # Это защитит от ошибки, если сменишь модель эмбеддингов
    vector_size = len(embeddings.embed_query("test"))

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    vectorstore.add_documents(chunks)
    print(f"Проиндексировано чанков: {len(chunks)} в коллекцию {collection_name}")

    return vectorstore


def delete_collection(collection_name: str):
    """
    Удаляет коллекцию из Qdrant.
    """
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"Коллекция {collection_name} удалена")
    else:
        print()

def get_collections():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    collections = client.get_collections()
    return collections


if __name__ == '__main__':
    delete_collection('corporate_docs')
