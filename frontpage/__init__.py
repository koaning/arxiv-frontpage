import json 
import itertools as it
from pathlib import Path
from functools import cached_property

import srsly
import questionary
from wasabi import Printer
from lazylines import LazyLines, read_jsonl

from ._download import main as _download

msg = Printer()

RAW_CONTENT_FILE = "raw/content.jsonl"
TRAINED_FOLDER_FOLDER = "training"

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
        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])

    @cached_property
    def model(self):
        from ._model import SentenceModel
        return SentenceModel.from_disk(TRAINED_FOLDER_FOLDER, encoder=self.encoder)

    def to_sentence_examples(self, stream, tag):
        for ex in stream:
            base = {"meta": ex["meta"], "label": tag}
            yield {"text": ex["title"], **base}
            for sent in self.nlp(ex["abstract"]).sents:
                yield {"text": sent.text, **base}

    def raw_content_stream(self):
        return srsly.read_jsonl(RAW_CONTENT_FILE)

    def tag_content_stream(self, tag):
        assert tag in self.tags
        glob = Path("downloads").glob("**/*.jsonl")
        full_data = it.chain(*list(srsly.read_jsonl(file) for file in glob))
        return (item for item in dedup_stream(full_data) if item['meta']['tag'] == tag)

    def tag_content_path(self, tag):
        return Path("raw") / f"{tag}.jsonl"

    def index(self):
        """Index annotation examples for quick annotation."""
        from simsity import create_index

        for tag in self.tags:
            msg.text(f"Setting up index for tag: {tag}", color="cyan")
            stream = (read_jsonl(self.tag_content_path(tag))
                      .progress()
                      .map(lambda d: d['text']))
            create_index(list(stream), self.encoder, path=Path("indices") / tag)

    def preprocess(self, index=True):
        glob = Path("downloads").glob("**/*.jsonl")
        msg.text("Generating raw/content.jsonl file.", color="cyan")
        full_stream = it.chain(*(srsly.read_jsonl(file) for file in glob))
        stream = LazyLines(full_stream).progress().pipe(dedup_stream)
        srsly.write_jsonl(RAW_CONTENT_FILE, stream)
        for tag in self.tags:
            msg.text(f"Generating data for {tag}.", color="cyan")
            stream = (read_jsonl(RAW_CONTENT_FILE)
                      .progress()
                      .keep(lambda d: d['meta']['tag'] == tag)
                      .pipe(self.to_sentence_examples, tag=tag))
            srsly.write_jsonl(self.tag_content_path(tag), stream)
        if index:
            self.index()
    
    def teams_create(self):
        for section in self.sections:
            stream_file = f"raw/{section['tag']}.jsonl"
            cloud_path_prefix = "{assets}/vdw/"
            asset_name = '"' + str(section['name']) + '"'
            asset_cmd = "ptc assets create "
            asset_cmd += asset_name + " "
            asset_cmd += cloud_path_prefix + stream_file + " "
            meta = {"loader": "jsonl"}
            asset_cmd += "--kind input --meta " + f"'{json.dumps(meta)}'"
            print(asset_cmd)
            
            file_cmd = "ptc files cp -r "
            file_cmd += f"{stream_file} "
            file_cmd += f"{cloud_path_prefix}/{stream_file} --make-dirs --overwrite"
            print(file_cmd)

            task_cmd = "ptc tasks create textcat "
            task_cmd += f"frontpage-{section['tag']} "
            task_cmd += f"--dataset frontpage-{section['tag']} "
            task_cmd += f"--input {asset_name} --label {section['tag']} "
            print(task_cmd)
            print(" ")
            # f372f8bc-f4a4-47b2-b055-44e7723a01bc

    def annotate(self):
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
        model.to_disk(TRAINED_FOLDER_FOLDER)
        # print(model("download my stuff from github yo"))
        # loaded = SentenceModel.from_disk(TRAINED_FOLDER_FOLDER, encoder=SentenceEncoder())
        # print(loaded("download my stuff from github yo"))
    
    def evaluate(self):
        ...

    def push_wandb(self):
        ...

    def pull_wandb(self):
        ...
