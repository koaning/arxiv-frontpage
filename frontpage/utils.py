from rich.console import Console 
import itertools as it

from spacy.tokens import Span


console = Console()


def batched(iterable, n=56):
    "Batch data into tuples of length n. The last batch may be shorter."
    if n < 1:
        raise ValueError('n must be at least one')
    iters = iter(iterable)
    while batch := tuple(it.islice(iters, n)):
        yield batch


def dedup_stream(stream, key="text"):
    uniq = {}
    for ex in stream:
        uniq[hash(ex[key])] = ex
    for ex in uniq.values():
        yield ex


def add_rownum(stream):
    for i, ex in enumerate(stream):
        yield {"text": ex["text"], "idx": i}


def attach_docs(lines, nlp, label):
    tuples = ((eg['text'], eg) for eg in lines)
    for doc, eg in nlp.pipe(tuples, as_tuples=True):
        eg['doc'] = sentence_classifier(doc, eg['preds'], label)
        yield eg


def sentence_classifier(doc, preds, label):
    doc.spans["sc"] = []
    for sent, pred in zip(doc.sents, preds):
        for k, p in pred.items():
            if p >= 0.6:
                if k == label:
                    doc.spans["sc"].append(Span(doc, sent.start, sent.end, k))
                    doc.cats[k] = max(doc.cats.get(k, 0.0), p)
    return doc


def attach_spans(stream, label, min_spans=1, max_spans=1):
    for ex in stream:
        spans = []
        for spansvals in ex['doc'].spans.values():
            for span in spansvals:
                spans.append(
                    {
                        "token_start": span.start,
                        "token_end": span.end - 1,
                        "start": span.start_char,
                        "end": span.end_char,
                        "text": span.text,
                        "label": label,
                    }
                )
        ex["spans"] = spans
        del ex["doc"]
        if len(spans) >= min_spans:
            if len(spans) <= max_spans:
                yield ex


def add_predictions(stream, model):
    for ex in stream:
        preds = model.predict(ex['sentences'])
        ex['preds'] = preds
        ex['created'] = ex['created'][:10]
        yield ex


def _abstract_single_annot_to_sent(example, nlp, label):
    """Takens an annotation from abstract level and turns it into a training example"""
    text = example['text']
    if example['answer'] == "accept" and "spans" in example:
        for span in example['spans']:
            yield {"text": text[span['start']: span['end']], label: 1}
        for span in example['spans']:
            text = text.replace(text[span['start']: span['end']], "")
        for sent in nlp(text).sents:
            if len(sent.text) > 5:
                yield {"text": sent.text, label: 0}


def abstract_annot_to_sent(examples, nlp, label):
    """Takens an annotation from abstract level and turns it into a training example"""
    for ex in examples:
        for annot in _abstract_single_annot_to_sent(ex, nlp, label):
            yield annot
