from typing import Dict, List
import random
import json
import itertools as it
from pathlib import Path 
from functools import cached_property

import srsly
from wasabi import Printer
from lazylines import LazyLines
from lunr import lunr
from lunr.index import Index

from .pipeline import dedup_stream, add_rownum, attach_spans, attach_docs
from .constants import DATA_LEVELS, INDICES_FOLDER, LABELS, CONFIG, THRESHOLDS, CLEAN_DOWNLOADS_FOLDER, DOWNLOADS_FOLDER
from .modelling import SentenceModel
from .utils import console

msg = Printer()


class DataStream:
    def __init__(self) -> None:
        pass

    @cached_property
    def db(self):
        from prodigy.components.db import connect
        
        db = connect()
        return db
    
    def get_dataset_name(self, label:str, level:str):
        """Source of truth as far as dataset name goes."""
        return f"{label}-{level}"
    
    def retreive_dataset_names(self):
        """Retreive the dataset names that actually have annotated data."""
        product = it.product(LABELS, DATA_LEVELS)
        possible = [self.get_dataset_name(lab, lev) for lab,lev in product]
        return [n for n in possible if n in self.db.datasets]
    
    def get_raw_download_stream(self):
        # Fetch all downloaded files, make sure most recent ones come first
        glob = reversed(list(DOWNLOADS_FOLDER.glob("**/*.jsonl")))
        
        # Make lazy generator for all the items
        stream = it.chain(*list(srsly.read_jsonl(file) for file in glob))
        return stream
    
    def save_clean_download_stream(self):
        stream = dedup_stream(self.get_raw_download_stream(), key="abstract")
        nested = LazyLines(stream).mutate(created=lambda d: d['created'][:10]).nest_by("created")
        for group in nested:
            CLEAN_DOWNLOADS_FOLDER.mkdir(parents=True, exist_ok=True)
            filepath = CLEAN_DOWNLOADS_FOLDER / f"{group['created']}.jsonl"
            g = ({**ex, "created": group['created']} for ex in group['subset'])
            srsly.write_jsonl(filepath, g)
        console.log(f"Cleaned files written in [bold]{CLEAN_DOWNLOADS_FOLDER}[/bold] folder.")
    
    def get_clean_download_stream(self):
        # Fetch all downloaded files, make sure most recent ones come first
        glob = reversed(list(CLEAN_DOWNLOADS_FOLDER.glob("**/*.jsonl")))
        
        # Make lazy generator for all the items
        stream = it.chain(*list(srsly.read_jsonl(file) for file in glob))
        return stream
    
    def get_download_stream(self, level:str="sentence"):
        """Stream of downloaded data, ready for annotation"""
        # Start out with the raw stream
        stream = self.get_clean_download_stream()
        
        # Generate two streams lazily
        abstract_stream = ({"text": ex["abstract"], "meta": {"url": ex["url"], "title": ex["title"], "created": ex["created"][:10]}} 
                           for ex in stream)
        sentences_stream = ({"text": sent, "meta": {"url": ex["url"]}} 
                            for ex in stream for sent in ex['sentences'])
        stream = abstract_stream if level == "abstract" else sentences_stream
        return stream
    
    def _sentence_data_to_train_format(self, stream):
        """Data ready for training from a sentence-level dataset."""
        for ex in stream:
            # This bit of logic ensures we ignore the `ignore` answer
            outcome = None
            if ex["answer"] == "accept":
                outcome = 1
            if ex["answer"] == "reject":
                outcome = 0
            if outcome is not None:
                yield {
                    "text": ex["text"],
                    ex["label"]: outcome
                }
    
    def _accumulate_train_stream(self, stream) -> List[Dict]:
        """
        This function ensures that we have each `text` appear only
        once and that the categories are nested in the `cats` key.
        """
        return (LazyLines(stream)
                .nest_by("text")
                .mutate(cats = lambda d: {k: v for ex in d['subset'] for k, v in ex.items()})
                .drop("subset")
                .collect())

    def get_train_stream(self) -> List[Dict]:
        console.log("Generating training stream")
        stream = []
        for dataset in self.retreive_dataset_names():
            if "sentence" in dataset:
                datapoints = self.db.get_dataset_examples(dataset)
                stream.extend(self._sentence_data_to_train_format(datapoints))
            # TODO: also handle the abstract level examples
        return self._accumulate_train_stream(stream)

    def get_lunr_stream(self, query: str, level: str):
        idx_path = self._index_path(kind="lunr", level=level)

        with open(idx_path) as fd:
            reloaded = json.loads(fd.read())
        idx = Index.load(reloaded)
        documents = (LazyLines(self.get_download_stream(level=level))
                     .pipe(add_rownum)
                     .collect())
        return [documents[int(i['ref'])] for i in idx.search(query)]

    def get_ann_stream(self, query: str, level: str):
        from simsity import load_index
        model = SentenceModel().from_disk()
        idx = load_index(self._index_path(kind="simsity", level=level), encoder=model.encoder)
        texts, scores = idx.query([query], n=150)
        for txt, score in zip(texts, scores):
            example = {"text": txt}
            example["meta"] = {"distance": float(score)}
            yield example

    def get_active_learn_stream(self, label, preference):
        from prodigy import set_hashes
        stream = self.get_download_stream(level="sentence")
        model = SentenceModel.from_disk()
    
        def make_scored_stream(stream, model):
            for ex in stream: 
                ex = set_hashes(ex)
                score = model(ex['text'])[label]
                ex['meta']['score'] = score
                yield score, ex 
            
        scored_stream = make_scored_stream(stream, model)
        if preference == "uncertainty":
            return (ex for s, ex in scored_stream if s < 0.6 and s > 0.4)
        if preference == "positive class":
            return (ex for s, ex in scored_stream if s > 0.6)
        if preference == "negative class":
            return (ex for s, ex in scored_stream if s < 0.4)

    def get_second_opinion_stream(self, model: SentenceModel, label, min_sents=1, max_sents=1):
        from prodigy.components.preprocess import add_tokens
        
        stream = self.content_stream(view="abstract")
        stream = ({'abstract': ex['text'], **ex} for ex in stream)
        model = SentenceModel.from_disk("training", encoder=SentenceModel().encoder)
        stream = attach_docs(stream, model.nlp, model=model)
        stream = attach_spans(stream, label, min_spans=min_sents, max_spans=max_sents)
        return add_tokens(model.nlp, stream)

    def get_random_stream(self, level:str):
        return (ex for ex in self.get_download_stream(level=level) if random.random() < 0.05)

    def _index_path(self, kind:str, level:str) -> Path:
        """kind is lunr vs. simsity, level is sentence vs. abstract"""
        path = INDICES_FOLDER / kind / level
        if kind == "simsity":
            return path
        path = Path(f"{path}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def create_indices(self, model: SentenceModel):
        """Index annotation examples for quick annotation."""
        from simsity import create_index

        for level in DATA_LEVELS:
            stream = LazyLines(self.get_download_stream(level=level)).map(lambda d: d['text']).collect()
            msg.info(f"Creating indices for {len(stream)} items.")
            msg.info(f"Preparing simsity index for {level}")
            path = self._index_path(kind="simsity", level=level)
            create_index(stream, model.encoder, path=path, batch_size=200, pbar=True)

            msg.info(f"Preparing lunr index for {level}")
            stream = LazyLines(self.get_download_stream(level=level)).pipe(add_rownum).collect()
            index = lunr(ref='idx', fields=('text',), documents=stream)
            serialized = index.serialize()
            with open(self._index_path(kind="lunr", level=level), 'w') as fd:
                json.dump(serialized, fd)

    def show_annot_stats(self):
        """Show the annotation statistics."""
        for level in DATA_LEVELS:
            data = {}
            for label in LABELS:
                dataset_name = f"{label}-{level}"
                if dataset_name in self.db.datasets:
                    examples = self.db.get_dataset_examples(dataset_name)
                    data[dataset_name] = [
                        dataset_name,
                        sum(1 for ex in examples if ex['answer'] == 'accept'),
                        sum(1 for ex in examples if ex['answer'] == 'ignore'),
                        sum(1 for ex in examples if ex['answer'] == 'reject')
                    ]
            msg.table(data.values(), 
                    header=["label", "accept", "ignore", "reject"], 
                    divider=True, 
                    aligns="r,r,r,r".split(","))
    
    def get_site_stream(self):
        model = SentenceModel.from_disk()

        def upper_limit(stream):
            tracker = {lab: 0 for lab in LABELS}
            limit = 35
            for ex in stream:
                for preds in ex['preds']:
                    for name, proba in preds.items():
                        if name in tracker and proba > THRESHOLDS[name] and tracker[name] < limit:
                            tracker[name] += 1
                            if "sections" not in ex:
                                ex['sections'] = []
                            ex['sections'].append(name)
                            ex['sections'] = list(set(ex['sections']))
                            yield ex
                if all(v == limit for v in tracker.values()):
                    break

        def add_predictions(stream, model: SentenceModel):
            for ex in stream:
                preds = model.predict(ex['sentences'])
                ex['preds'] = preds
                ex['created'] = ex['created'][:10]
                yield ex

        console.log("Filtering recent content.")
        return (
            LazyLines(self.get_clean_download_stream())
                .head(1000)
                .pipe(add_predictions, model=model)
                .pipe(upper_limit)
                .collect()
        )
    
    def get_site_content(self):
        site_stream = self.get_site_stream()
        sections = {dict(section)['label']: {**dict(section), "content": []} for section in CONFIG.sections}

        def render_html(item, section):
            text = ""
            for sent, pred in zip(item['sentences'], item['preds']):
                proba = pred[section]
                addition = sent
                if proba > THRESHOLDS[section]:
                    addition = f"<span class='px-1 mx-1 bg-yellow-200'>{addition}</span>"
                text += addition
            return f"<p>{text}</p>"

        for item in site_stream:
            for section in item['sections']:
                item['html'] = render_html(item, section)
                sections[section]['content'].append(item)

        console.log("Sections generated.")
        return list(sections.values())
        
