import srsly
import numpy as np
from embetter.text import SentenceEncoder
from sklearn.linear_model import LogisticRegression
from wasabi import Printer
from prodigy.components.db import connect
from icepickle.linear_model import save_coefficients, load_coefficients
from pathlib import Path


msg = Printer()


class SentenceModel:
    def __init__(self, encoder, tasks):
        self._encoder = encoder
        self._tasks = tasks
        self._models = {k: LogisticRegression() for k in self._tasks}

    def update(self, examples):
        X = self._encoder.transform([ex["text"] for ex in examples])
        for task, model in self._models.items():
            xs = np.array([X[i] for i, ex in enumerate(examples) if task in ex])
            ys = np.array(
                [ex[task] for i, ex in enumerate(examples) if task in ex], dtype=int
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

    def performance(self, valid_examples):
        pass

    def to_disk(self, path):
        for name, clf in self._models.items():
            save_coefficients(clf, Path(path) / f"{name}.h5")

    @classmethod
    def from_disk(cls, path, encoder):
        models = {}
        for f in Path(path).glob("*.h5"):
            clf_reloaded = LogisticRegression()
            load_coefficients(clf_reloaded, f)
            models[f.stem] = clf_reloaded

        model = SentenceModel(encoder=encoder, tasks=models.keys())
        model._models = models
        return model


if __name__ == "__main__":
    db = connect()
    config = srsly.read_yaml("config.yml")

    train_data = {}
    for section in config["sections"]:
        tag = section["tag"]
        if tag in db.datasets:
            for ex in db.get_dataset_examples(tag):
                if ex["answer"] != "ignore":
                    h = ex["_input_hash"]
                    if h not in train_data:
                        train_data[h] = {"text": ex["text"]}
                    train_data[h][tag] = int(ex["answer"] == "accept")

    train_data = train_data.values()

    tasks = [s["tag"] for s in config["sections"] if s["tag"] in db.datasets]
    model = SentenceModel(encoder=SentenceEncoder(), tasks=tasks)
    model.update(train_data)
    model.to_disk("training")
    print(model("download my stuff from github yo"))
    loaded = SentenceModel.from_disk("training", encoder=SentenceEncoder())
    print(loaded("download my stuff from github yo"))
