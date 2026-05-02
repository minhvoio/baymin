from pydantic import BaseModel
from typing import Literal, Optional


class AnswerStructure(BaseModel):
    answer: str

class QuizAnswer(BaseModel):
    A_or_B_or_C_or_D: Literal['A', 'B', 'C', 'D']