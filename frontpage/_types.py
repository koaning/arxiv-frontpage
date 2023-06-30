from typing import List
from pydantic import BaseModel 


class ArxivArticle(BaseModel):
    created: str
    title: str
    abstract: str
    sentences: List[str]
    url: str
