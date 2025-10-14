from pydantic import BaseModel
from typing import Literal, Optional


class AnswerStructure(BaseModel):
    answer: str


class AnswerWithThinking(BaseModel):
    answer: str
    thinking: Optional[str] = None


class QuizAnswer(BaseModel):
    A_or_B_or_C_or_D: Literal['A', 'B', 'C', 'D']


