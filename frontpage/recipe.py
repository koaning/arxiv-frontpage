import random 
from pathlib import Path 


import prodigy
from wasabi import Printer
from frontpage import Frontpage
from prodigy import set_hashes

msg = Printer()


fp = Frontpage.from_config_file("config.yml")


def get_stream_index(label:str, setting:str):
    from simsity import load_index
    idx = load_index(Path("indices") / label, encoder=fp.encoder)
    texts, scores = idx.query(setting, n=150)
    for txt, score in zip(texts, scores):
        example = {"text": txt}
        example = set_hashes(example)
        example["meta"] = {"score": score}
        yield example


def get_stream_random(label: str):
    stream = fp.raw_content_stream()
    return (ex for ex in fp.to_sentence_examples(stream, tag=label) if random.random() < 0.05)


def get_stream_active_learn(label: str, setting:str):
    from ._model import SentenceModel
    from prodigy.components.sorters import prefer_uncertain, prefer_high_scores, prefer_low_scores

    stream = fp.tag_content_stream(tag=label)
    stream = fp.to_sentence_examples(stream, tag=label)
    model = SentenceModel.from_disk("training", encoder=fp.encoder)
    
    def make_scored_stream(stream, model):
        for ex in stream: 
            ex = set_hashes(ex)
            score = model(ex['text'])[label]
            if 'meta' not in ex:
                ex['meta'] = {}
            ex['meta']['score'] = score
            yield score, ex 
        
    scored_stream = make_scored_stream(stream, model)
    if setting == "uncertainty":
        return prefer_uncertain(scored_stream)
    if setting == "positive class":
        return prefer_high_scores(((s, ex) for s, ex in scored_stream if s > 0.5))
    if setting == "negative class":
        return prefer_low_scores(((s, ex) for s, ex in scored_stream if s < 0.5))


@prodigy.recipe("textcat.arxiv.sentence",
    label=("The label to annotate", "positional", None, str),
    tactic=("The tactic to retreive relevant examples", "positional", None, str),
    setting=("Additional setting for the tactic", "positional", None, str),
)
def arxiv_sentence(label, tactic, setting):
    """Very general recipe to annotate sentences, using different ordering techniques."""
    if tactic == "simsity":
        msg.info("Setting up simsity stream")
        stream = get_stream_index(label, setting)
    elif tactic == "random":
        msg.info("Setting up randomized stream")
        stream = get_stream_random(label)
    elif tactic == "active-learning":
        msg.info("Setting up active learning")
        stream = get_stream_active_learn(label, setting)
    else:
        raise ValueError("This should never happen.")
    
    return {
        "dataset": label,
        "stream": ({**ex, "label": label} for ex in stream),
        "view_id": "classification",
    }
