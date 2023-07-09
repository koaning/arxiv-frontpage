from tqdm.rich import tqdm
import jinja2 
import datetime as dt 
import json 
import itertools as it
from pathlib import Path
from functools import cached_property

import srsly
import questionary
from lunr import lunr
from wasabi import Printer
from lazylines import LazyLines, read_jsonl
from embetter.utils import cached
import warnings
from tqdm import TqdmExperimentalWarning

from .pipeline import dedup_stream, add_rownum, attach_docs
from .datastream import DataStream

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

msg = Printer()

TRAINED_FOLDER_FOLDER = "training"
TEMPLATE_PATH = "templates/home.html"
CONFIG_FILE = "config.yml"
DATA_LEVELS = ["sentence", "abstract"]


class Frontpage:
    """This is the main object that contains all project verbs."""

    def __init__(self):
        self.config = srsly.read_yaml(CONFIG_FILE)
        self.sections = self.config["sections"]
        self.labels = [s["label"] for s in self.config["sections"]]
        self.datastream = DataStream()

    @cached_property
    def encoder(self):
        from embetter.text import SentenceEncoder
        encoder = SentenceEncoder()
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

    def _dataset_name(self, label:str, view:str) -> str:
        return f"{view}-{label}"
    
    @property
    def _annotation_views(self):
        return ["sentence", "abstract"]

    def content_stream(self, view:str):
        # Fetch all downloaded files, make sure most recent ones come first
        glob = reversed(list(Path("downloads").glob("**/*.jsonl")))
        # Make lazy generator for all the items
        stream = it.chain(*list(srsly.read_jsonl(file) for file in glob))
        # Generate two streams lazily
        abstract_stream = ({"text": ex["abstract"], "meta": {"url": ex["url"], "title": ex["title"], "created": ex["created"][:10]}} 
                           for ex in stream)
        sentences_stream = ({"text": sent, "meta": {"url": ex["url"]}} 
                            for ex in stream for sent in ex['sentences'])
        return dedup_stream(abstract_stream) if view == "abstract" else dedup_stream(sentences_stream)

    def index(self):
        """Index annotation examples for quick annotation."""
        from simsity import create_index

        for view in self._annotation_views:
            msg.info(f"Preparing simsity index for {view}")
            stream = (LazyLines(self.content_stream(view=view)).map(lambda d: d['text']))
            create_index(list(stream), self.encoder, path=self._index_path(kind="simsity", view=view), pbar=True)

            msg.info(f"Preparing lunr index for {view}")
            stream = (LazyLines(self.content_stream(view=view)).pipe(add_rownum))
            index = lunr(ref='idx', fields=('text',), documents=list(stream))
            serialized = index.serialize()
            with open(self._index_path(kind="lunr", view=view), 'w') as fd:
                json.dump(serialized, fd)


    def annotate(self):
        """
        Methods for abstract level.
            - second opinion
            - simsity
            - search-engine
            - random

        Methods for sentence level.
            - filter by patterns
            - active learn by sentence
            - simsity
            - search-engine
            - random
        """
        

        from .recipe import arxiv_sentence, arxiv_abstract
        from prodigy.app import server 
        from prodigy.core import Controller

        dataset_name = f"{results['label']}-{results['view']}"
        name = "textcat.arxiv.sentence" if results['view'] == 'sentence' else "textcat.arxiv.abstract"
        if results['view'] == 'sentence':
            ctrl_data = arxiv_sentence(dataset_name, results['label'], results['tactic'], results['setting'])
        else:
            ctrl_data = arxiv_abstract(dataset_name, results['label'], results['tactic'], results['setting'])
        controller = Controller.from_components(name, ctrl_data)
        server(controller, controller.config)   

    @cached_property
    def db(self):
        from prodigy.components.db import connect
        
        db = connect()
        return db

    def fetch_annotated_data(self):
        train_data = {}
        found_tags = []
        for tag in self.tags:
            tag = f"{tag}-sentence"
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

        # TODO: 
        # so yeah, this can be made way faster. we're looping over the same 
        # thing again and again here, which is wasteful. we should also introcuce
        # a max lookup to the

        model = SentenceModel.from_disk(TRAINED_FOLDER_FOLDER, encoder=self.encoder)
        stream = self.content_stream(view="abstract")

        def render_html(doc):
            text = doc.text
            for span in doc.spans["sc"]:
                text = text.replace(span.text, f"<span style='background-color: rgb(254 240 138);'>{span.text}</span>")
            return f"<p>{text}</p>"

        return (LazyLines(stream)
            .mutate(abstract=lambda d: d['text'])
            .pipe(attach_docs, nlp=self.nlp, model=model)
            .mutate(cats = lambda d: d['doc'].cats)
            .keep(lambda d: d['cats'].get(tag, 0.0) > 0.6)
            .mutate(html=lambda d: render_html(d['doc']),
                    n_sents=lambda d: len(d['doc'].spans["sc"]),
                    link=lambda d: d['meta']['url']))

    def build(self):
        config = srsly.read_yaml(CONFIG_FILE)
        for section in tqdm(config['sections'], desc="Looping over tags."):
            print(section['name'])
            section["content"] = self.fetch_tag_candidate_stream(tag=section['tag']).head(20).collect()
        import pprint
        pprint.pprint(section["content"])
        
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
