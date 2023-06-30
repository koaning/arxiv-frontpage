import datetime as dt 

import tqdm
import srsly
import arxiv
from pathlib import Path
from arxiv import Result
import spacy
from spacy.language import Language
from ._types import ArxivArticle


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
        title=res.title,
        abstract=summary,
        sentences=sents,
        url=res.entry_id
    )


def main():
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])

    items = arxiv.Search(
        query="and",
        max_results=500,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    results = list(items.results())

    articles = [parse(r, nlp=nlp) 
                for r in tqdm.tqdm(results) 
                if total_seconds(r) < 2.5 and r.primary_category.startswith("cs")]

    filename = str(dt.datetime.now()).replace(" ", "-")[:13] + "h.jsonl"
    srsly.write_jsonl(Path("downloads") / filename, [dict(a) for a in articles])
