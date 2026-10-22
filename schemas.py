from pydantic import BaseModel, Field


class TextInput(BaseModel):
    question: str = Field(min_length=10)
