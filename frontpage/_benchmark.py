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
from sklearn.pipeline import make_pipeline, make_union
from sklearn.decomposition import TruncatedSVD
from embetter.finetune import ForwardFinetuner, ContrastiveFinetuner
from sklearn.preprocessing import FunctionTransformer

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
    "cohere": CohereEncoder(),
}

encoders["multi"] = make_union(
    encoders["sbert"], 
    make_pipeline(
        HashingVectorizer(n_features=10_000), 
        TruncatedSVD(),
    )
)

tuners = {
    "forward": lambda: ForwardFinetuner(hidden_dim=300), 
    "contrast": lambda: ContrastiveFinetuner(hidden_dim=300),
    "none": lambda: FunctionTransformer()
}

for name, enc in encoders.items():
    if name not in ["multi", "hash_lg", "hash_sm"]:
        encoders[name] = cached(f"cache/{str(type(enc))}", enc)

models = {
    "logistic": LogisticRegression(class_weight="balanced", max_iter=1000), 
    "svm": SVC(class_weight="balanced")
}

def calc_stats(pred_valid, y_valid):
    return {**classification_report(pred_valid, y_valid, output_dict=True)['1'],  "accuracy": float(np.mean(pred_valid == y_valid))}


def run_benchmark(label, model, encoder, tuner):
    res = {"label": label, "model": model, "encoder": encoder, "tuner": tuner}
    pipe = make_pipeline(encoders[encoder], tuners[tuner](), models[model])
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

        probas = pipe.predict_proba(X_valid)
        for thres in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
            valid_pred = (probas[:, 1] > thres).astype(int)
            
            stats = calc_stats(valid_pred, y_valid)
            res = {**res, **stats, "data_size": len(y), "i": i, "threshold": float(thres)}
            yield res



if __name__ == "__main__":
    settings = grid(
        label=["new-dataset"], 
        encoder=["sbert", "openai", "cohere", "multi"], 
        model=["logistic", "svm"],
        tuner=["contrast", "forward", "none"]
    )

    stats = (ex for setting in settings for ex in run_benchmark(**setting))

    Path("benchmark.jsonl").unlink()
    srsly.write_jsonl("benchmark.jsonl", stats)

    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_width_chars(1000)

    print(
        pl.read_ndjson("benchmark.jsonl")
        .groupby("label","model","encoder","tuner")
        .agg(
            pl.mean("recall"), 
            pl.mean("precision"), 
            pl.mean("f1-score"),
            pl.mean("accuracy"),
            pl.mean("infer_time")
        ).sort("f1-score")
    )

