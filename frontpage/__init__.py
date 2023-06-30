from tqdm.rich import tqdm
import jinja2 
import datetime as dt 
import json 
import itertools as it
from pathlib import Path
from functools import cached_property

from spacy.tokens import Span

import srsly
import questionary
from wasabi import Printer
from lazylines import LazyLines, read_jsonl
from embetter.utils import cached
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

from frontpage._download import main as _download

msg = Printer()

RAW_CONTENT_FILE = "raw/content.jsonl"
TRAINED_FOLDER_FOLDER = "training"
TEMPLATE_PATH = "templates/home.html"
CONFIG_FILE = "config.yml"

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

    @cached_property
    def encoder(self):
        from embetter.text import SentenceEncoder
        from embetter.external import OpenAIEncoder, CohereEncoder
        if self.config['encoder'] == "sbert":
            encoder = SentenceEncoder()
        if self.config['encoder'] == "openai":
            encoder = OpenAIEncoder()
        if self.config['encoder'] == "cohere":
            encoder = CohereEncoder()
        encoder = cached(f"cache/{str(type(encoder))}", encoder)
        return encoder

    @cached_property
    def nlp(self):
        import spacy
        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])

    @cached_property
    def model(self):
        from ._model import SentenceModel
        return SentenceModel.from_disk(TRAINED_FOLDER_FOLDER, encoder=self.encoder)

    def to_sentence_examples(self, stream, tag:str):
        for ex in stream:
            base = {"meta": ex.get("meta", {}), "label": tag}
            yield {"text": ex["title"], **base}
            for sent in self.nlp(ex["abstract"]).sents:
                yield {"text": sent.text, **base}

    def raw_content_stream(self):
        return srsly.read_jsonl(RAW_CONTENT_FILE)

    def tag_content_stream(self, tag:str):
        assert tag in self.tags
        glob = reversed(list(Path("downloads").glob("**/*.jsonl")))
        full_data = it.chain(*list(srsly.read_jsonl(file) for file in glob))
        return (item for item in dedup_stream(full_data) if item['meta']['tag'] == tag)

    def tag_content_path(self, tag:str):
        return Path("raw") / f"{tag}.jsonl"

    def index(self):
        """Index annotation examples for quick annotation."""
        from simsity import create_index

        for tag in self.tags:
            stream = (read_jsonl(self.tag_content_path(tag))
                      .progress(f"Encoding examples for tag: {tag:>15}")
                      .map(lambda d: d['text']))
            create_index(list(stream), self.encoder, path=Path("indices") / tag, pbar=False)

    def preprocess(self, index=True):
        glob = Path("downloads").glob("**/*.jsonl")
        full_stream = it.chain(*(srsly.read_jsonl(file) for file in glob))
        stream = (LazyLines(full_stream)
                  .pipe(dedup_stream))
        srsly.write_jsonl(RAW_CONTENT_FILE, stream)
        for tag in self.tags:
            stream = (read_jsonl(RAW_CONTENT_FILE)
                      .progress(desc=f"Generating data for {tag:>15}.")
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

    def annotate(self):
        results = {}
        results["label"] = questionary.select(
            "Which label do you want to annotate?",
            choices=self.tags,
        ).ask()

        results["tactic"] = questionary.select(
            "Which tactic do you want to apply?",
            choices=["simsity", "random", "active-learning", "second-opinion"],
        ).ask()

        results['setting'] = ''
        if results["tactic"] == "simsity":
            results["setting"] = questionary.text(
                "What query sentence would you like to use?", ""
            ).ask()

        if results["tactic"] == "active-learning":
            results["setting"] = questionary.select(
                "What should the active learning method prefer?",
                choices=["positive class", "uncertainty", "negative class"],
            ).ask()

        from .recipe import arxiv_sentence
        from prodigy.app import server 
        from prodigy.core import Controller

        ctrl_data = arxiv_sentence(results['label'], results['tactic'], results['setting'])
        controller = Controller.from_components("textcat.arxiv.sentence", ctrl_data)
        server(controller, controller.config)

    def show_annot_stats(self):
        """Show the annotation statistics."""
        data = {}
        for tag in self.tags:
            if tag in self.db.datasets:
                examples = self.db.get_dataset_examples(tag)
                data[tag] = [
                    tag,
                    sum(1 for ex in examples if ex['answer'] == 'accept'),
                    sum(1 for ex in examples if ex['answer'] == 'ignore'),
                    sum(1 for ex in examples if ex['answer'] == 'reject')
                ]
        msg.table(data.values(), 
                  header=["label", "accept", "ignore", "reject"], 
                  divider=True, 
                  aligns="r,r,r,r".split(","))

    def gridsearch(self):
        """Show the annotation statistics."""
        pass

    @cached_property
    def db(self):
        from prodigy.components.db import connect
        
        db = connect()
        return db

    def fetch_annotated_data(self):
        train_data = {}
        found_tags = []
        for tag in self.tags:
            if tag in self.db.datasets:
                if len(self.db.get_dataset_examples(tag)) == 0:
                    msg.warn(f"Skipping training for {tag}. No training examples.")
                else:
                    msg.info(f"Preparing data for {tag}.")
                    found_tags.append(tag)
                    for ex in self.db.get_dataset_examples(tag):
                        if ex["answer"] != "ignore":
                            h = ex["_input_hash"]
                            if h not in train_data:
                                train_data[h] = {"text": ex["text"]}
                            train_data[h][tag] = int(ex["answer"] == "accept")

        return train_data.values(), found_tags
        
    
    def train(self):
        from ._model import SentenceModel

        annotated_data, found_tags = self.fetch_annotated_data()

        model = SentenceModel(encoder=self.encoder, tasks=found_tags)
        model.update(annotated_data)
        model.to_disk(TRAINED_FOLDER_FOLDER)
    
    def fetch_tag_candidate_stream(self, tag:str):
        from frontpage._model import SentenceModel

        model = SentenceModel.from_disk(TRAINED_FOLDER_FOLDER, encoder=self.encoder)
        stream = self.tag_content_stream(tag=tag)

        def attach_docs(lines, nlp):
            tuples = ((eg['abstract'], eg) for eg in lines)
            for doc, eg in nlp.pipe(tuples, as_tuples=True):
                eg['doc'] = sentence_classifier(doc)
                yield eg

        def sentence_classifier(doc):
            doc.spans["sc"] = []
            for sent in doc.sents:
                preds = model(sent.text)
                for k, p in preds.items():
                    if p >= 0.6:
                        doc.spans["sc"].append(Span(doc, sent.start, sent.end, k))
                        doc.cats[k] = max(doc.cats.get(k, 0.0), p)
            return doc

        def render_html(doc):
            text = doc.text
            for span in doc.spans["sc"]:
                text = text.replace(span.text, f"<span style='background-color: rgb(254 240 138);'>{span.text}</span>")
            return f"<p>{text}</p>"

        return (LazyLines(stream)
            .pipe(attach_docs, nlp=self.nlp)
            .mutate(cats = lambda d: d['doc'].cats)
            .keep(lambda d: d['cats'].get(tag, 0.0) > 0.6)
            .mutate(html=lambda d: render_html(d['doc']),
                    n_sents=lambda d: len(d['doc'].spans["sc"]),
                    link=lambda d: d['meta']['link']))

    def build(self):
        config = srsly.read_yaml(CONFIG_FILE)
        for section in tqdm(config['sections'], desc="Looping over tags."):
            section["content"] = self.fetch_tag_candidate_stream(tag=section['tag']).head(20).collect()
        
        template = jinja2.Template(Path(TEMPLATE_PATH).read_text())
        Path("site.html").write_text(template.render(sections=config['sections'], today=dt.date.today()))

    
    def evaluate(self):
        from ._benchmark import benchmark
        annotated, found_tags = self.fetch_annotated_data()
        benchmark(annotated, tags=["new-dataset", "data-quality"])

    def push_wandb(self):
        ...

    def pull_wandb(self):
        ...


if __name__ == "__main__":
    Frontpage(config=srsly.read_yaml("config.yml")).build()
