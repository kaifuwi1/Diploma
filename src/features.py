# src/features.py
import re, string, nltk, spacy
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download("vader_lexicon", quiet=True)
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
sia = SentimentIntensityAnalyzer()

class HandCraftedFeaturizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def _pos_ratio(self, doc, pos):
        return sum(1 for t in doc if t.pos_ == pos) / max(len(doc), 1)

    def transform(self, texts):
        feats = []
        for text in texts:
            doc = nlp(text.lower())

            words = [t.text for t in doc if not t.is_punct]
            n_chars = len(text)
            n_words = len(words)

            feat = dict(
                n_chars=n_chars,
                n_words=n_words,
                avg_word_len=np.mean([len(w) for w in words]) if words else 0,
                pct_capitals=sum(1 for c in text if c.isupper()) / max(n_chars, 1),
                num_exclaim=text.count("!"),
                num_qmark=text.count("?"),
                vader_neg=sia.polarity_scores(text)["neg"],
                vader_pos=sia.polarity_scores(text)["pos"],
                pos_noun=self._pos_ratio(doc, "NOUN"),
                pos_verb=self._pos_ratio(doc, "VERB"),
                pos_adj=self._pos_ratio(doc, "ADJ"),
                pos_adv=self._pos_ratio(doc, "ADV"),
            )
            feats.append(feat)
        return pd.DataFrame(feats)
