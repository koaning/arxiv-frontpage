import itertools as it
from pathlib import Path

import srsly
import questionary
from wasabi import Printer

from ._download import main as _download

msg = Printer()


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

    @property
    def encoder(self):
        from embetter.text import SentenceEncoder

        return SentenceEncoder()

    def index(self):
        """Index annotation examples for quick annotation."""
        import spacy
        from simsity import create_index

        nlp = spacy.load("en_core_web_sm")

        def to_examples(stream, nlp, tag):
            for ex in stream:
                if ex["meta"]["tag"] == tag:
                    base = {"meta": ex["meta"], "label": tag}
                    yield {"text": ex["title"], **base}
                    for sent in nlp(ex["abstract"]).sents:
                        yield {"text": sent.text, **base}

        for tag in self.tags:
            msg.text(f"Setting up index for tag: {tag}", color="cyan")
            stream = srsly.read_jsonl("raw/content.jsonl")
            stream = (ex["text"] for ex in to_examples(stream, nlp=nlp, tag=tag))
            create_index(list(stream), self.encoder, path=Path("indices") / tag)

    def preprocess(self):
        glob = Path("downloads").glob("**/*.jsonl")
        full_data = it.chain(*list(srsly.read_jsonl(file) for file in glob))
        stream = (item for item in dedup_stream(full_data))
        srsly.write_jsonl("raw/content.jsonl", stream)
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

        if results["tactic"] == "simsity":
            results["setting"] = questionary.text(
                "What query sentence would you like to use?", ""
            ).ask()

        if results["tactic"] == "active-learning":
            results["setting"] = questionary.select(
                "What should the active learning method prefer?",
                choices=["uncertainty", "positive class", "negative class"],
            ).ask()

        return results

    def annotate(self):
        print(self.run_questionaire())

    def train(self):
        ...

    def annot(self):
        # Start up a query that launched Prodigy with questionary
        ...

    def evaluate(self):
        ...

    def push_wandb(self):
        ...

    def pull_wandb(self):
        ...
