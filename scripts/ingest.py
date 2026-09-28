from rag.ingestion import (
    load_all_documents,
    split_documents,
    enrich_metadata,
    index_documents,
)

from dotenv import load_dotenv

load_dotenv()


def main():
    print("Загрузка документов...")
    documents = load_all_documents("data/raw")

    print(f"Загружено документов: {len(documents)}")

    if len(documents) == 0:
        print("Чанков нет. Индексация не запускается.")
        return

    print("Разбиение на чанки...")
    chunks = split_documents(documents)

    print(f"Получено чанков: {len(chunks)}")

    print("Обогащение метаданных...")
    chunks = enrich_metadata(chunks)

    print("Индексация в Qdrant...")
    index_documents(chunks)

    print("Готово.")

if __name__ == "__main__":
    main()