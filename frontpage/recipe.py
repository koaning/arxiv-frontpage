import spacy
import questionary

import prodigy
from wasabi import Printer

from .datastream import DataStream
from .constants import LABELS, DATA_LEVELS
datastream = DataStream()

msg = Printer()


def run_questions():
    results = {}
    results["label"] = questionary.select(
        "Which label do you want to annotate?",
        choices=LABELS,
    ).ask()

    results["level"] = questionary.select(
        "What view of the data do you want to take?",
        choices=DATA_LEVELS,
    ).ask()

    if results["level"] == "abstract":
        choices = ["second-opinion", "search-engine", "simsity", "random"]
    else:
        choices = ["simsity", "search-engine", "active-learning", "random"]

    results["tactic"] = questionary.select(
        "Which tactic do you want to apply?",
        choices=choices,
    ).ask()

    results['setting'] = ''
    if results["tactic"] in ["simsity", "search-engine"]:
        results["setting"] = questionary.text(
            "What query would you like to use?", ""
        ).ask()

    if results["tactic"] == "active-learning":
        results["setting"] = questionary.select(
            "What should the active learning method prefer?",
            choices=["positive class", "uncertainty", "negative class"],
        ).ask()
    return results 


@prodigy.recipe("textcat.arxiv.sentence",
    dataset=("The dataset to save", "positional", None, str),
    label=("The label to annotate", "positional", None, str),
    tactic=("The tactic to retreive relevant examples", "positional", None, str),
    setting=("Additional setting for the tactic", "positional", None, str),
)
def arxiv_sentence(dataset, label, tactic, setting):
    """Very general recipe to annotate sentences, using different ordering techniques."""
    from prodigy import set_hashes
    if tactic == "simsity":
        msg.info("Setting up simsity stream")
        stream = datastream.get_ann_stream(query=setting, level="sentence")
    elif tactic == "random":
        msg.info("Setting up randomized stream")
        stream = datastream.get_random_stream(level="sentence")
    elif tactic == "active-learning":
        msg.info("Setting up active learning")
        stream = datastream.get_active_learn_stream(label=label, preference=setting)
    elif tactic == "search-engine":
        msg.info("Setting up lunr query")
        stream = datastream.get_lunr_stream(query=setting, level="sentence")
    else:
        raise ValueError("This should never happen.")
    
    return {
        "dataset": dataset,
        "stream": (set_hashes({**ex, "label": label}) for ex in stream),
        "view_id": "classification",
        "config":{
            "exclude_by": "input"
        }
    }


@prodigy.recipe("textcat.arxiv.abstract",
    dataset=("The dataset to save", "positional", None, str),
    label=("The label to annotate", "positional", None, str),
    tactic=("The tactic to retreive relevant examples", "positional", None, str),
    setting=("Additional setting for the tactic", "positional", None, str),
)
def arxiv_abstract(dataset, label, tactic, setting):
    """Very general recipe to annotate sentences, using different ordering techniques."""
    from prodigy.components.preprocess import add_tokens

    if tactic == "simsity":
        msg.info("Setting up simsity stream")
        stream = datastream.get_ann_stream(query=setting, level="abstract")
    elif tactic == "random":
        msg.info("Setting up randomized stream")
        stream = datastream.get_random_stream(level="abstract")
    elif tactic == "search-engine":
        msg.info("Setting up lunr query")
        stream = datastream.get_lunr_stream(query=setting, level="sentence")
    elif tactic == "second-opinion":
        msg.info("Setting up second opinion")
        stream = datastream.get_second_opinion_stream(label=label, min_sents=1, max_sents=2)
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
            ],
            "exclude_by": "input"
        }
    }


def annotate_prodigy():
    from prodigy.app import server 
    from prodigy.core import Controller
    
    results = run_questions()

    dataset_name = datastream.get_dataset_name(results['label'], results['level'])
    name = "textcat.arxiv.sentence" if results['level'] == 'sentence' else "textcat.arxiv.abstract"
    if results['level'] == 'sentence':
        ctrl_data = arxiv_sentence(dataset_name, results['label'], results['tactic'], results['setting'])
    else:
        ctrl_data = arxiv_abstract(dataset_name, results['label'], results['tactic'], results['setting'])
    controller = Controller.from_components(name, ctrl_data)
    server(controller, controller.config)
