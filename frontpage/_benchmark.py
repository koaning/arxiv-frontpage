import time
from pathlib import Path
import itertools as it 

import tqdm
import srsly
import numpy as np
import polars as pl
from dotenv import load_dotenv
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from embetter.text import SentenceEncoder, spaCyEncoder
from embetter.external import CohereEncoder, OpenAIEncoder
from embetter.utils import cached

from frontpage.datastream import DataStream


load_dotenv()

def grid(**kwargs):
    res = [{k: v for k, v in zip(kwargs.keys(), prod)} 
            for prod in it.product(*[v for v in kwargs.values()])]
    return tqdm.tqdm(res)
        

datastream = DataStream()

k_folder = StratifiedKFold(n_splits=10)

encoders = {
    "spacy": spaCyEncoder("en_core_web_md"),
    "sbert": SentenceEncoder(),
    "hash_lg": HashingVectorizer(),
    "hash_sm": HashingVectorizer(n_features=2**14),
    "openai": OpenAIEncoder(),
    "cohere": CohereEncoder()
}

for name, enc in encoders.items():
    if "hash" not in name:
        encoders[name] = cached(f"cache/{str(type(enc))}", enc)

models = {
    "logistic": LogisticRegression(class_weight="balanced", max_iter=1000), 
    "svm": SVC(class_weight="balanced")
}

def calc_stats(pred_valid, y_valid):
    return {**classification_report(pred_valid, y_valid, output_dict=True)['1'],  "accuracy": float(np.mean(pred_valid == y_valid))}

def run_benchmark(label, model, encoder):
    res = {"label": label, "model": model, "encoder": encoder}
    pipe = make_pipeline(encoders[encoder] ,models[model])
    examples = datastream.get_train_stream()
    X = [ex['text'] for ex in examples if label in ex['cats']]
    y = [ex['cats'][label] for ex in examples if label in ex['cats']]
    folds = k_folder.split(X, y)
    for i, (train_idx, valid_idx) in enumerate(folds):
        X_train = [str(x) for x in np.array(X)[train_idx]]
        X_valid = [str(x) for x in np.array(X)[valid_idx]]
        y_train = np.array(y)[train_idx]
        y_valid = np.array(y)[valid_idx]
        pipe.fit(X_train, y_train)

        tic = time.time()
        valid_pred = pipe.predict(X_valid)
        toc = time.time()
        
        stats = calc_stats(valid_pred, y_valid)
        yield {**res, "i": i, "n_infer": len(y_valid), "infer_time": toc - tic, **stats}

if __name__ == "__main__":
    settings = grid(
        label=["new-dataset"], 
        encoder=["hash_sm", "hash_lg", "spacy", "sbert", "cohere", "openai"], 
        model=["logistic", "svm"]
    )

    stats = (ex for setting in settings for ex in run_benchmark(**setting))

    Path("benchmark.jsonl").unlink()
    srsly.write_jsonl("benchmark.jsonl", stats)

    pl.Config.set_tbl_rows(100)

    print(
        pl.read_ndjson("benchmark.jsonl")
        .groupby("label","model","encoder")
        .agg(
            pl.mean("recall"), 
            pl.mean("precision"), 
            pl.mean("f1-score"),
            pl.mean("accuracy"),
            pl.mean("infer_time")
        ).sort("f1-score")
    )

