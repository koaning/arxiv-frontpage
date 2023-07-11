import time 
import itertools as it 
from tqdm.rich import tqdm 

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import classification_report
from embetter.text import SentenceEncoder, spaCyEncoder
# from embetter.external import CohereEncoder, OpenAIEncoder
from embetter.utils import cached

def calc_stats(pred_valid, y_valid):
    return classification_report(pred_valid, y_valid, output_dict=True)['1']

def benchmark_tag(annot_data, tag, resample=10):
    relevant_data = [ex for ex in annot_data if tag in ex]
    texts = [ex['text'] for ex in relevant_data]
    labels = [ex[tag] for ex in relevant_data]
    encoders = {
        "spacy": spaCyEncoder("en_core_web_md"),
        "sbert": SentenceEncoder(),
        "hash": HashingVectorizer()
    }
    for name, enc in encoders.items():
        if name not in ["cv", "hash"]:
            encoders[name] = cached(f"cache/{str(type(enc))}", enc)
    models = {
        "logistic": LogisticRegression(class_weight="balanced", max_iter=1000), 
        "svm": SVC(class_weight="balanced")
    }
    grid = list(it.product(encoders.keys(), models.keys(), range(resample)))
    
    dataset = []
    for (enc, mod, r) in tqdm(grid):
        X_train, X_valid, y_train, y_valid = train_test_split(texts, labels, test_size=0.5, random_state=r)
        pipe = make_pipeline(encoders[enc], models[mod])
        tic = time.time()
        pipe.fit(X_train, y_train)
        time.time() - tic
        tic = time.time()
        pred_train = pipe.predict(X_train)
        time.time() - tic
        pred_valid = pipe.predict(X_valid)
        accuracy_train = np.mean(pred_train == y_train)
        accuracy_valid = np.mean(pred_valid == y_valid)
        dataset.append({
            "tag": tag,
            "acc_train": float(accuracy_train), 
            "acc_valid": float(accuracy_valid),
            "enc": enc,
            "mod": mod,
            "random_seed": r,
            **calc_stats(pred_valid, y_valid)
        })
    return dataset

def benchmark(annot_data, tags):
    results = []
    for tag in tags:
        new_results = benchmark_tag(annot_data, tag)
        results.extend(new_results)
        df = (pl.DataFrame(new_results)
            .groupby(pl.col("tag"), pl.col("enc"), pl.col("mod"))
            .agg(
                pl.mean("acc_train"),
                pl.mean("acc_valid"),
                pl.mean("precision"),
                pl.mean("recall"),
                pl.mean("f1-score"),
            )
            .sort("tag", "enc", "mod", descending=True))
        print(df)
