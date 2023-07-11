"""
This script is pretty much the same thing as `python -m frontpage download`. 
It's just that this script adds a retry mechanic and has less dependencies
so that it's more lightweight to run from Github Actions.
"""

import logging
import datetime as dt 
from pathlib import Path
from typing import List

import srsly
import tqdm
import arxiv
from arxiv import Result
from retry import retry 
import spacy
from spacy.language import Language
from pydantic import BaseModel 


class ArxivArticle(BaseModel):
    created: str
    title: str
    abstract: str
    sentences: List[str]
    url: str


def total_seconds(res: Result) -> float:
    """Get total seconds from now from Arxiv result"""
    now = dt.datetime.now(dt.timezone.utc)
    return (now - res.published).total_seconds() / 3600 / 24


def parse(res: Result, nlp: Language) -> ArxivArticle:
    """Parse proper Pydantic object from Arxiv"""
    summary = res.summary.replace("\n", " ")
    doc = nlp(summary)
    sents = [s.text for s in doc.sents]
    
    return ArxivArticle(
        created=str(res.published)[:19], 
        title=str(res.title),
        abstract=summary,
        sentences=sents,
        url=str(res.entry_id)
    )

@retry(tries=5, delay=1, backoff=2)
def main():
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])

    items = arxiv.Search(
        query="and",
        max_results=500,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    results = list(items.results())

    articles = [dict(parse(r, nlp=nlp)) 
                for r in tqdm.tqdm(results) 
                if total_seconds(r) < 2.5 and r.primary_category.startswith("cs")]

    filename = str(dt.datetime.now()).replace(" ", "-")[:13] + "h.jsonl"
    srsly.write_jsonl(Path("downloads") / filename, articles)


if __name__ == "__main__":
    logging.basicConfig()
    main()
