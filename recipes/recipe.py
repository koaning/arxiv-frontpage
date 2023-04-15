from pathlib import Path 

from prodigy import recipe 

import prodigy
import spacy
import srsly
import random 
from simsity import create_index, load_index

from prodigy import set_hashes
from prodigy.components.loaders import JSONL
from prodigy.components.filters import filter_tasks
from prodigy.components.db import connect

from embetter.text import SentenceEncoder, spaCyEncoder
from sklearn.pipeline import make_pipeline
from wasabi import Printer

msg = Printer()


class KeyGrabber():
    """
    Effectively the same thing as the ColumnGrabber, except this is
    meant to work on generators of dictionaries instead of dataframes.
    """

    def __init__(self, colname: str) -> None:
        self.colname = colname

    def transform(self, X, y=None):
        """
        Takes a column from pandas and returns it as a list.
        """
        if isinstance(X, dict):
            return X[self.colname]
        return [x[self.colname] for x in X]


def setup_everything(tag):
    config = srsly.read_yaml("config.yml")
    tags = [s['tag'] for s in config['sections']]
    if tag not in tags:
        raise ValueError(f"The {tag} tag is not defined in any section in config.yml")

    nlp = spacy.load("en_core_web_md")
    db = connect()

    # set up stream and set hashes
    stream = (
        item for item in JSONL("raw/content.jsonl") if item["meta"]["tag"] == tag
    )

    stream = filter_tasks(stream, db.get_task_hashes(tag))
    return stream, nlp
    


def to_examples(stream, nlp, tag):
    for ex in stream:
        base = {"meta": ex["meta"], "label": tag}
        yield {"text": ex['title'], **base}
        for sent in nlp(ex['abstract']).sents:
            yield {"text": sent.text, **base}


@prodigy.recipe("textcat.index")
def index():
    config = srsly.read_yaml("config.yml")
    encoder = SentenceEncoder()
    for section in config['sections']:
        tag = section['tag']
        msg.text(f"Setting up index for tag: {tag}", color="cyan")

        stream, nlp = setup_everything(tag)
        stream = (ex['text'] for ex in to_examples(stream, nlp=nlp, tag=tag))

        create_index(list(stream), encoder, path=Path("indices") / tag)


@prodigy.recipe("textcat.from-index",
    tag=("The tag to annotate", "positional", None, str),
    query=("The text query to use", "positional", None, str),
)
def from_index(tag, query):
    setup_everything(tag)
    encoder = SentenceEncoder()
    index = load_index(Path("indices") / tag, encoder=encoder)
    examples, dist = index.query(query, n=150)

    def add_label(stream):
        for ex in stream:
            yield {
                "text": ex['data'],
                "label": tag
            }
    
    stream = (set_hashes(ex) for ex in add_label(examples))
    print(next(stream))
    
    return {
        "dataset": tag,
        "stream": stream,
        "view_id": "classification",
    }


@prodigy.recipe(
    "textcat.sentence",
    tag=("The tag to annotate", "positional", None, str),
)
def textcat(tag):
    # import spaCy and initialize matcher
    stream, nlp = setup_everything(tag)
    stream = to_examples(stream, nlp=nlp, tag=tag)

    stream = (set_hashes(ex, input_keys=("text"), task_keys=("label")) for ex in stream)

    return {
        "dataset": tag,
        "stream": stream,
        "view_id": "classification",
    }
