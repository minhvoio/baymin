from pydantic import BaseModel
from typing import Literal, Optional


class AnswerStructure(BaseModel):
    answer: str


class AnswerWithThinking(BaseModel):
    answer: str
    thinking: Optional[str] = None


class QuizAnswer(BaseModel):
    one_letter_answer: Literal['A', 'B', 'C', 'D']


