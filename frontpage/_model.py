from .utils import batched
from spacy.tokens import Span
import numpy as np
from sklearn.linear_model import LogisticRegression
from wasabi import Printer
from skops.io import dump, load
from pathlib import Path
msg = Printer() 


class SentenceModel:
    def __init__(self, encoder, tasks):
        self._encoder = encoder
        self._tasks = tasks
        self._models = {k: LogisticRegression(class_weight="balanced") for k in self._tasks}

    def update(self, examples):
        X = self._encoder.transform([ex["text"] for ex in examples])
        for task, model in self._models.items():
            xs = np.array([X[i] for i, ex in enumerate(examples) if task in ex])
            ys = np.array(
                [ex[task] for ex in examples if task in ex], dtype=int
            )
            model.fit(xs, ys)
            msg.good(f"Trained the {task} task, using {len(xs)} examples.")

    def __call__(self, text):
        result = {}
        X = self._encoder.transform([text])
        for task in self._tasks:
            proba = self._models[task].predict_proba(X)[0, 1]
            result[task] = float(proba)
        return result

    def predict(self, stream):
        for batch in batched(stream):
            texts = [ex['text'] for ex in batch]
            X = self._encoder.transform(texts)
            preds = {}
            for task in self._tasks:
                preds[task] = list(self._models[task].predict_proba(texts))
            for i, ex in enumerate(batch):
                yield {**ex, "cats": {task: preds[task][i] for task in self._tasks}}
                

    def to_disk(self, path):
        for name, clf in self._models.items():
            dump(clf, Path(path) / f"{name}.h5")

    @classmethod
    def from_disk(cls, path: Path, encoder):
        if not Path(path).exists():
            raise RuntimeError("You need to train a model beforehand.")
        models = {}
        for f in Path(path).glob("*.h5"):
            models[f.stem] = load(f, trusted=True)

        model = SentenceModel(encoder=encoder, tasks=models.keys())
        model._models = models
        return model


def sentence_classifier(doc, model):
    doc.spans["sc"] = []
    for sent in doc.sents:
        preds = model(sent.text)
        for k, p in preds.items():
            if p >= 0.6:
                doc.spans["sc"].append(Span(doc, sent.start, sent.end, k))
                doc.cats[k] = max(doc.cats.get(k, 0.0), p)
    return doc

def attach_docs(lines, nlp, model):
    tuples = ((eg['abstract'], eg) for eg in lines)
    for doc, eg in nlp.pipe(tuples, as_tuples=True):
        eg['doc'] = sentence_classifier(doc, model)
        yield eg


def render_html(doc):
    text = doc.text
    for span in doc.spans["sc"]:
        text = text.replace(span.text, f"<span style='background-color: rgb(254 240 138);'>{span.text}</span>")
    return f"<p>{text}</p>"