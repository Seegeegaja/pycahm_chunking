# 각 QA를 구성하는 클래스
from typing import List
from pydantic import BaseModel

class QA(BaseModel):
    question: str
    answer: str
#     QA 전체를 담는 클레스
class Output(BaseModel):
    qa_list: List[QA]