import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag.ingestion import (
    enrich_metadata,
    index_documents,
    load_all_documents,
    split_documents,
)

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def main():
    path_data = "data/raw/"

    project_root = Path(__file__).resolve().parents[1]
    final_path = project_root.joinpath(path_data)

    logger.debug(f"path: {final_path}")
    documents = load_all_documents(str(final_path))

    logger.info(f"Загружено документов: {len(documents)}")

    if len(documents) == 0:
        logger.info("Чанков нет. Индексация не запускается.")
        return

    logger.info("Разбиение на чанки...")
    chunks = split_documents(documents)

    logger.info(f"Получено чанков: {len(chunks)}")

    logger.info("Обогащение метаданных...")
    chunks = enrich_metadata(chunks)

    logger.info("Индексация в Qdrant...")
    index_documents(chunks)

    logger.info("Готово.")

if __name__ == "__main__":
    main()