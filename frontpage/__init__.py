import itertools as it
from pathlib import Path
from functools import cached_property

import srsly
import questionary
from wasabi import Printer

from ._download import main as _download

msg = Printer()

RAW_CONTENT_FILE = "raw/content.jsonl"

def dedup_stream(stream):
    uniq = {}
    for ex in stream:
        uniq[hash(ex["title"])] = ex
    for ex in uniq.values():
        yield ex


class Frontpage:
    """This is the main object that contains all project verbs."""

    def __init__(self, config):
        self.config = config
        self.sections = self.config["sections"]
        self.tags = [s["tag"] for s in self.config["sections"]]

    @classmethod
    def from_config_file(cls, path):
        """Loads from a config file."""
        return cls(config=srsly.read_yaml(path))

    def download(self):
        """Download new data for today."""
        _download(self.config)

    @cached_property
    def encoder(self):
        from embetter.text import SentenceEncoder
        return SentenceEncoder()

    @cached_property
    def nlp(self):
        import spacy
        return spacy.load("en_core_web_sm")

    def to_sentence_examples(self, stream, tag):
        for ex in stream:
            if ex["meta"]["tag"] == tag:
                base = {"meta": ex["meta"], "label": tag}
                yield {"text": ex["title"], **base}
                for sent in self.nlp(ex["abstract"]).sents:
                    yield {"text": sent.text, **base}

    def raw_content_stream(self):
        return srsly.read_jsonl(RAW_CONTENT_FILE)

    def index(self):
        """Index annotation examples for quick annotation."""
        from simsity import create_index

        for tag in self.tags:
            msg.text(f"Setting up index for tag: {tag}", color="cyan")
            stream = srsly.read_jsonl(RAW_CONTENT_FILE)
            stream = (ex["text"] for ex in self.to_sentence_examples(stream, nlp=nlp, tag=tag))
            create_index(list(stream), self.encoder, path=Path("indices") / tag)

    def preprocess(self):
        glob = Path("downloads").glob("**/*.jsonl")
        full_data = it.chain(*list(srsly.read_jsonl(file) for file in glob))
        stream = (item for item in dedup_stream(full_data))
        srsly.write_jsonl(RAW_CONTENT_FILE, stream)
        msg.text("Created raw/content.jsonl file.", color="cyan")
        self.index()

    def run_questionaire(self):
        results = {}
        results["label"] = questionary.select(
            "Which label do you want to annotate?",
            choices=self.tags,
        ).ask()

        results["tactic"] = questionary.select(
            "Which tactic do you want to apply?",
            choices=["simsity", "random", "active-learning"],
        ).ask()

        results['setting'] = ''
        if results["tactic"] == "simsity":
            results["setting"] = questionary.text(
                "What query sentence would you like to use?", ""
            ).ask()

        if results["tactic"] == "active-learning":
            results["setting"] = questionary.select(
                "What should the active learning method prefer?",
                choices=["uncertainty", "positive class", "negative class"],
            ).ask()

        from .recipe import arxiv_sentence
        from prodigy.app import server 
        from prodigy.core import Controller

        ctrl_data = arxiv_sentence(results['label'], results['tactic'], results['setting'])
        controller = Controller.from_components("textcat.arxiv.sentence", ctrl_data)
        server(controller, controller.config)

    def annotate(self):
        results = self.run_questionaire()
        print(results)

    def show_annot_stats(self):
        """Show the annotation statistics."""
        pass

    def gridsearch(self):
        """Show the annotation statistics."""
        pass

    def train(self):
        from ._model import SentenceModel
        from prodigy.components.db import connect
        
        db = connect()

        train_data = {}
        for tag in self.tags:
            if tag in db.datasets:
                for ex in db.get_dataset_examples(tag):
                    if ex["answer"] != "ignore":
                        h = ex["_input_hash"]
                        if h not in train_data:
                            train_data[h] = {"text": ex["text"]}
                        train_data[h][tag] = int(ex["answer"] == "accept")

        train_data = train_data.values()

        tasks = [s["tag"] for s in self.sections if s["tag"] in db.datasets]
        model = SentenceModel(encoder=self.encoder, tasks=tasks)
        model.update(train_data)
        model.to_disk("training")
        # print(model("download my stuff from github yo"))
        # loaded = SentenceModel.from_disk("training", encoder=SentenceEncoder())
        # print(loaded("download my stuff from github yo"))

    def evaluate(self):
        ...

    def push_wandb(self):
        ...

    def pull_wandb(self):
        ...
