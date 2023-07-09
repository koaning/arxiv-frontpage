import json
import itertools as it
from pathlib import Path 
from functools import cached_property

import srsly
from wasabi import Printer
from lazylines import LazyLines
from lunr import lunr
from lunr.index import Index

from frontpage.pipeline import dedup_stream, add_rownum
from frontpage.constants import DATA_LEVELS, INDICES_FOLDER, LABELS
from frontpage.modelling import SentenceModel

msg = Printer()


class DataStream:
    def __init__(self) -> None:
        pass

    @cached_property
    def db(self):
        from prodigy.components.db import connect
        
        db = connect()
        return db
    
    
    def get_download_stream(self, level:str="sentence"):
        # Fetch all downloaded files, make sure most recent ones come first
        glob = reversed(list(Path("downloads").glob("**/*.jsonl")))
        
        # Make lazy generator for all the items
        stream = it.chain(*list(srsly.read_jsonl(file) for file in glob))
        
        # Generate two streams lazily
        abstract_stream = ({"text": ex["abstract"], "meta": {"url": ex["url"], "title": ex["title"], "created": ex["created"][:10]}} 
                           for ex in stream)
        sentences_stream = ({"text": sent, "meta": {"url": ex["url"]}} 
                            for ex in stream for sent in ex['sentences'])
        stream = abstract_stream if level == "abstract" else sentences_stream
        return dedup_stream(stream)

    def get_train_stream(task:str):
        pass

    def get_lunr_stream(query: str, level: str, task: str):
        pass

    def get_ann_stream(query: str, level: str, task: str):
        pass

    def get_active_learn_stream(model, preference):
        pass

    def get_second_opinion_stream(model, preference):
        pass

    def get_random_stream(level:str):
        pass

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