import json 
import spacy
import random 
from pathlib import Path 


import prodigy
from lazylines import LazyLines
from wasabi import Printer
from . import Frontpage
from ._pipeline import add_rownum, attach_docs
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
    from ._model import SentenceModel
    stream = fp.content_stream(view="abstract")
    stream = ({'abstract': ex['text'], **ex} for ex in stream)
    model = SentenceModel.from_disk("training", encoder=fp.encoder)
    stream = attach_docs(stream, fp.nlp, model=model)

    def attach_spans(stream):
        for ex in stream:
            spans = []
            for spansvals in ex['doc'].spans.values():
                for span in spansvals:
                    spans.append(
                        {
                            "token_start": span.start,
                            "token_end": span.end - 1,
                            "start": span.start_char,
                            "end": span.end_char,
                            "text": span.text,
                            "label": label,
                        }
                    )
            ex["spans"] = spans
            del ex["doc"]
            if len(spans) >= 1:
                print(ex)
                yield ex
    
    return add_tokens(fp.nlp, attach_spans(stream))


def get_stream_active_learn(view:str, label: str, setting:str):
    from ._model import SentenceModel
    from prodigy.components.sorters import prefer_uncertain

    stream = fp.content_stream(view=view)
    model = SentenceModel.from_disk("training", encoder=fp.encoder)
    
    def make_scored_stream(stream, model):
        for ex in stream: 
            ex = set_hashes(ex)
            score = model(ex['text'])[label]
            ex['meta']['score'] = score
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
        stream = get_stream_active_learn("sentence", label, setting)
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
        stream = get_stream_active_learn("abstract", label, setting)
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
