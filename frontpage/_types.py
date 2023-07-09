from typing import List
from pydantic import BaseModel 


class ArxivArticle(BaseModel):
    created: str
    title: str
    abstract: str
    sentences: List[str]
    url: str


class LabelConfig(BaseModel):
    name: str
    instructions: str
    label: str
    threshold: float = 0.7


class Config(BaseModel):
    n_per_section: int = 20
    sections: List[LabelConfig]
