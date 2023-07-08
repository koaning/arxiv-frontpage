import json 
import spacy
import random 
from pathlib import Path 


import prodigy
from lazylines import LazyLines
from wasabi import Printer
from . import Frontpage
from ._pipeline import add_rownum
from prodigy import set_hashes
from prodigy.components.preprocess import add_tokens

msg = Printer()


fp = Frontpage()


def get_stream_index(view:str, setting:str):
    from simsity import load_index
    idx = load_index(fp._index_path(kind="simsity", view=view), encoder=fp.encoder)
    texts, scores = idx.query([setting], n=150)
    for txt, score in zip(texts, scores):
        example = {"text": txt}
        example["meta"] = {"distance": float(score)}
        yield example


def get_stream_random(label: str):
    stream = fp.tag_content_stream(tag=label)
    return (ex for ex in fp.to_sentence_examples(stream, tag=label) if random.random() < 0.05)


def get_stream_lunr(view:str, setting:str):
    from lunr.index import Index

    idx_path = fp._index_path(kind="lunr", view=view)

    with open(idx_path) as fd:
        reloaded = json.loads(fd.read())
    idx = Index.load(reloaded)
    documents = (LazyLines(fp.content_stream(view=view)).pipe(add_rownum).collect())
    return [documents[int(i['ref'])] for i in idx.search(setting)]


def get_stream_second_opinion(label: str):
    stream = fp.fetch_tag_candidate_stream(tag=label)
    stream = (ex for ex in stream if len(ex['doc'].spans) == 1)
    return (ex for ex in fp.to_sentence_examples(stream, tag=label))


def get_stream_active_learn(label: str, setting:str):
    from ._model import SentenceModel
    from prodigy.components.sorters import prefer_uncertain

    stream = fp.tag_content_stream(tag=label)
    stream = fp.to_sentence_examples(stream, tag=label)
    model = SentenceModel.from_disk("training", encoder=fp.encoder)
    
    def make_scored_stream(stream, model):
        for ex in stream: 
            ex = set_hashes(ex)
            score = model(ex['text'])[label]
            yield score, ex 
        
    scored_stream = make_scored_stream(stream, model)
    if setting == "uncertainty":
        return prefer_uncertain(scored_stream)
    if setting == "positive class":
        return (ex for s, ex in scored_stream if s > 0.6)
    if setting == "negative class":
        return (ex for s, ex in scored_stream if s < 0.4)


@prodigy.recipe("textcat.arxiv.sentence",
    dataset=("The dataset to save", "positional", None, str),
    label=("The label to annotate", "positional", None, str),
    tactic=("The tactic to retreive relevant examples", "positional", None, str),
    setting=("Additional setting for the tactic", "positional", None, str),
)
def arxiv_sentence(dataset, label, tactic, setting):
    """Very general recipe to annotate sentences, using different ordering techniques."""
    if tactic == "simsity":
        msg.info("Setting up simsity stream")
        stream = get_stream_index(view="sentence", setting=setting)
    elif tactic == "random":
        msg.info("Setting up randomized stream")
        stream = get_stream_random(label)
    elif tactic == "active-learning":
        msg.info("Setting up active learning")
        stream = get_stream_active_learn(label, setting)
    elif tactic == "second-opinion":
        msg.info("Setting up second opinion")
        stream = get_stream_second_opinion(label)
    elif tactic == "search-engine":
        msg.info("Setting up lunr query")
        stream = get_stream_lunr(view="sentence", setting=setting)
    else:
        raise ValueError("This should never happen.")
    
    return {
        "dataset": dataset,
        "stream": (set_hashes({**ex, "label": label}) for ex in stream),
        "view_id": "classification",
    }


@prodigy.recipe("textcat.arxiv.abstract",
    dataset=("The dataset to save", "positional", None, str),
    label=("The label to annotate", "positional", None, str),
    tactic=("The tactic to retreive relevant examples", "positional", None, str),
    setting=("Additional setting for the tactic", "positional", None, str),
)
def arxiv_abstract(dataset, label, tactic, setting):
    """Very general recipe to annotate sentences, using different ordering techniques."""
    if tactic == "simsity":
        msg.info("Setting up simsity stream")
        stream = get_stream_index(view="abstract", setting=setting)
    elif tactic == "random":
        msg.info("Setting up randomized stream")
        stream = get_stream_random(label)
    elif tactic == "active-learning":
        msg.info("Setting up active learning")
        stream = get_stream_active_learn(label, setting)
    elif tactic == "second-opinion":
        msg.info("Setting up second opinion")
        stream = get_stream_second_opinion(label)
    elif tactic == "search-engine":
        msg.info("Setting up lunr query")
        stream = get_stream_lunr(view="abstract", setting=setting)
    else:
        raise ValueError("This should never happen.")
    
    nlp = spacy.blank("en")
    stream = ({**ex, "label": label} for ex in stream)
    stream = add_tokens(nlp, stream)
    return {
        "dataset": dataset,
        "stream": (set_hashes(ex) for ex in stream),
        "view_id": "blocks",
        "config": {
            "labels": [label],
            "blocks": [
                {"view_id": "ner_manual"},
            ]
        }
    }
