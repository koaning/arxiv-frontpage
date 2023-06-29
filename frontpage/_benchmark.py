import time 
import itertools as it 

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer

from embetter.text import SentenceEncoder, spaCyEncoder
# from embetter.external import CohereEncoder, OpenAIEncoder
from embetter.utils import cached


def benchmark(annot_data, tag, resample=5):
    relevant_data = [ex for ex in annot_data if tag in ex]
    texts = [ex['text'] for ex in relevant_data]
    labels = [ex[tag] for ex in relevant_data]
    encoders = {"sbert": SentenceEncoder()}
    for name, enc in encoders.items():
        encoders[name] = cached(f"cache/{str(type(enc))}", enc)
    models = {"logistic": LogisticRegression(class_weight="balanced", ), "svm": SVC(class_weight="balanced")}
    grid = it.product(encoders.keys(), models.keys(), range(resample))
    for (enc, mod, r) in grid:
        X_train, X_valid, y_train, y_valid = train_test_split(texts, labels, test_size=0.5, random_state=r)
        pipe = make_pipeline(encoders[enc], models[mod])
        tic = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - tic
        tic = time.time()
        pred_train = pipe.predict(X_train)
        infer_time = time.time() - tic
        pred_valid = pipe.predict(X_valid)
        accuracy_train = np.mean(pred_train == y_train)
        accuracy_valid = np.mean(pred_valid == y_valid)
        print({
            "tag": tag,
            "acc_train": accuracy_train, 
            "acc_valid": accuracy_valid,
            "enc": enc,
            "mod": mod,
            "random_seed": r
        })
