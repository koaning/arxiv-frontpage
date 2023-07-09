from spacy.tokens import Span


def dedup_stream(stream):
    uniq = {}
    for ex in stream:
        uniq[hash(ex["text"])] = ex
    for ex in uniq.values():
        yield ex

def add_rownum(stream):
    for i, ex in enumerate(stream):
        yield {"text": ex["text"], "idx": i}

def attach_docs(lines, nlp, model):
    tuples = ((eg['abstract'], eg) for eg in lines)
    for doc, eg in nlp.pipe(tuples, as_tuples=True):
        eg['doc'] = sentence_classifier(doc, model)
        yield eg

def sentence_classifier(doc, model):
    doc.spans["sc"] = []
    for sent in doc.sents:
        preds = model(sent.text)
        for k, p in preds.items():
            if p >= 0.6:
                doc.spans["sc"].append(Span(doc, sent.start, sent.end, k))
                doc.cats[k] = max(doc.cats.get(k, 0.0), p)
    return doc