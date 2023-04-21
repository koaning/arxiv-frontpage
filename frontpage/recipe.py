from pathlib import Path 


import prodigy
from wasabi import Printer
from frontpage import Frontpage

msg = Printer()


fp = Frontpage.from_config_file("config.yml")


def get_stream_index(label:str, setting:str):
    from simsity import load_index
    from prodigy import set_hashes
    idx = load_index(Path("indices") / label, encoder=fp.encoder)
    for txt in idx.query(setting, n=150)[0]:
        example = {"text": txt}
        yield set_hashes(example)


def get_stream_random(label: str):
    stream = fp.raw_content_stream()
    return fp.to_sentence_examples(stream, tag=label)


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
        pass
    else:
        raise ValueError("This should never happen.")
    
    return {
        "dataset": label,
        "stream": stream,
        "view_id": "classification",
    }
