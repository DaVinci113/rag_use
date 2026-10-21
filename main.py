from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from fastapi import FastAPI
from fastapi import Request
from contextlib import asynccontextmanager


app = FastAPI(
    title="RAG Chat Bot",
    description="RAG Chat Bot",
    version="1.0",
    # lifespan=lifespan,
)


@asynccontextmanager
async def lifespan():
    async with aiohttp.ClientSession() as session:
        app.state.session = session
        app.state.agent = Agent(session=session)
        yield

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/chat")
async def chat(request: Request):
    return {"message": f"{request.body}"}



if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
