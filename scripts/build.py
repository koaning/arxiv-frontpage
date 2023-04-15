import spacy
from spacy.tokens import Span
from spacy.language import Language

from embetter.text import SentenceEncoder

from train import SentenceModel

model = SentenceModel.from_disk("training", encoder=SentenceEncoder())

@Language.component("sentence_classifier")
def sentence_classifier(doc):
    doc.spans["sc"] = []
    for sent in doc.sents:
        preds = model(sent.text)
        for k, p in preds.items():
            if p >= 0.6:
                doc.spans["sc"].append(Span(doc, sent.start, sent.end, k))
                doc.cats[k] = max(doc.cats.get(k, 0.0), p)
    return doc

nlp = spacy.load("en_core_web_sm", disable=["ner"])
nlp.add_pipe("sentence_classifier", name="sentence_classifier", last=True)
doc = nlp("The code and the dataset can be found on our Github repo. This is comparable to the error reported on the commonly used 3DPW dataset. We further introduce a new metric, local movement, which describes the movement of the wrist and ankle joints in relation to the body. With this, we show that SportsPose contains more movement than the Human3.6M and 3DPW datasets in these extremum joints, indicating that our movements are more dynamic. The dataset with accompanying code can be downloaded from our website. We hope that SportsPose will allow researchers and practitioners to develop and evaluate more effective models for the analysis of sports performance and injury prevention. With its realistic and diverse dataset, SportsPose provides a valuable resource for advancing the state-of-the-art in pose estimation in sports.")

def render_html(doc):
    text = doc.text
    for span in doc.spans["sc"]:
        text = text.replace(span.text, f"<span style='background-color: rgb(254 240 138);'>{span.text}</span>")
    return f"<p>{text}</p>"

print(render_html(doc))