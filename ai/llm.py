import logging
import os
import sys

import aiohttp
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv()

def _get_env(name):
    values = os.getenv(name)
    if not values:
        raise RuntimeError(f"No {name} environment variable is set")
    return values



class Agent:
    def __init__(
            self,
            session,
    ):
        self.session = session
        self.api_host = _get_env("LLM_HOST")
        self.model = _get_env("LLM_MODEL")
        self.api_key = _get_env("OPENAI_API_KEY")
        self.timelimit = _get_env("LLM_TIMELIMIT")
        self.system_prompt = _get_env("SYSTEM_PROMPT")
        self.model_api_key = _get_env("MODEL_API_KEY")

    def _build_headers(self):
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.model_api_key}',
        }

    def _build_json(self, question, rag_response):
        final_prompt =f"""
        'Question': {question}
        'RAG_DATA': {rag_response}
"""

        return {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': final_prompt},
            ]
        }

    async def process(self, user_question, rag_response):

        json_data = self._build_json(user_question, rag_response)
        response_from_llm = await ai_response(
            session=self.session,
            url=self.api_host,
            headers=self._build_headers(),
            json=json_data,
        )
        return response_from_llm

async def ai_response(
        session,
        url,
        headers,
        json,
):
    try:
        async with session.post(url, headers=headers, json=json) as response:
            result = await response.json()
            logger.debug(f'Ответ LLM: {result}')
            return result
    except aiohttp.ClientError as e:
        logger.error(e)
        return e


if __name__ == '__main__':
    import asyncio

    user_text = "Когда выпущен самый новый телефон?"

    rag={
        'TELEjone': '20 june 2010',
        'Elefone': '22 june 2020'
    }

    async def test_launch_llm():
        async with aiohttp.ClientSession() as session:
            agent = Agent(session=session)
            result = await agent.process(user_question=user_text, rag_response=rag)
            print(result)

    asyncio.run(test_launch_llm())
