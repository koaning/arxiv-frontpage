from functools import cached_property
from embetter.utils import cached


class SentenceModel:
    def __init__(self) -> None:
        pass

    @cached_property
    def encoder(self):
        from embetter.text import SentenceEncoder
        encoder = SentenceEncoder()
        encoder = cached(f"cache/{str(type(encoder))}", encoder)
        return encoder

    @cached_property
    def nlp(self):
        import spacy
        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])