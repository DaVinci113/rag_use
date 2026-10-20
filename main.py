import os
import sys

from langchain_openai import ChatOpenAI
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv()

model = os.getenv("LLM_MODEL")
host = os.getenv("LLM_HOST")
key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(
    model=model,
    base_url=host,
    api_key=key,
)

response = chat.invoke("hello")
logger.debug(response)
