from typing import List, Dict
from pathlib import Path
from functools import cached_property

from wasabi import Printer
import numpy as np
from skops.io import dump, load
from embetter.utils import cached
from sklearn.linear_model import LogisticRegression

from .constants import TRAINED_FOLDER_FOLDER, LABELS
from .utils import console 

msg = Printer()


class SentenceModel:
    def __init__(self, labels=LABELS) -> None:
        self.labels = labels 
        self._models = {k: LogisticRegression(class_weight="balanced") for k in self.labels}

    def train(self, examples):
        X = self.encoder.transform([ex["text"] for ex in examples])
        for task, model in self._models.items():
            xs = np.array([X[i] for i, ex in enumerate(examples) if task in ex['cats']])
            ys = np.array(
                [ex['cats'][task] for ex in examples if task in ex['cats']], dtype=int
            )
            model.fit(xs, ys)
            console.log(f"Trained the [bold]{task}[/bold] task, using {len(xs)} examples.")
        return self

    def __call__(self, text:str) -> Dict:
        result = {}
        X = self.encoder.transform([text])
        for label in self.labels:
            proba = self._models[label].predict_proba(X)[0, 1]
            result[label] = float(proba)
        return result
    
    def predict(self, texts: List[str]) -> List[Dict]:
        X = self.encoder.transform(texts)
        result = [{} for _ in texts]
        for label in self.labels:
            probas = self._models[label].predict_proba(X)[:, 1]
            for i, proba in enumerate(probas):
                result[i][label] = float(proba)
        return result

    @cached_property
    def encoder(self):
        from embetter.text import SentenceEncoder
        encoder = SentenceEncoder()
        encoder = cached(f"cache/{str(type(encoder))}", encoder)
        return encoder
    
    def vectorizer(self):
        return 

    @cached_property
    def nlp(self):
        import spacy
        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])

    def to_disk(self, path: Path=TRAINED_FOLDER_FOLDER):
        for name, clf in self._models.items():
            dump(clf, Path(path) / f"{name}.h5")
        console.log(f"Model saved in folder: {path}")

    @classmethod
    def from_disk(cls, path: Path=TRAINED_FOLDER_FOLDER):
        if not Path(path).exists():
            raise RuntimeError("You need to train a model beforehand.")
        models = {}
        for f in Path(path).glob("*.h5"):
            models[f.stem] = load(f, trusted=True)

        model = SentenceModel(labels=models.keys())
        model._models = models
        console.log(f"Model loaded from: {path}")
        return model
