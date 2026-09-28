import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


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
print(response.content)
