"""
Routines for preprocessing texts.
"""

# standard library
import re
from string import punctuation

# wrangling
from spacy.lang.en.stop_words import STOP_WORDS as STOP_WORDS_EN
from spacy.lang.fr.stop_words import STOP_WORDS as STOP_WORDS_FR
from spacy.lang.es.stop_words import STOP_WORDS as STOP_WORDS_ES

STOP_WORDS = STOP_WORDS_EN | STOP_WORDS_FR | STOP_WORDS_ES


__all__ = [
    "replace_numbers",
    "preprocess_text",
    "preprocess_example",
]


def replace_numbers(text: str) -> str:
    """
    Replace all numbers in a text, including integers, floats and numbers with a thousand separator.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    text : str
        Input text with all numbers replaced.

    Examples
    --------
    >>> replace_numbers('The country created about 2,078 km of paved road between 2015-2020 (22%).')
    'The country created about NUM km of paved road between NUM-NUM (NUM%).'
    >>> replace_numbers('In 2018, 11.0 cases were registered against 7 in 2017.')
    'In NUM, NUM cases were registered against NUM in NUM.'
    """
    pattern = r"(\b\d+[\.\,]?\d*\b)"
    text = re.sub(pattern, "NUM", text)
    return text


def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Preprocess text by removing stop words, stripping punctuation and replacing numbers with a placeholder value.

    Parameters
    ----------
    text : str
        Input text to preprocess.
    remove_stopwords : bool, default=True
        If True, remove stop words in English, French and Spanish.

    Returns
    -------
    text : str
        Preprocessed text.
    """
    text = text.replace("\ufffe", "")  # replace '￾', e.g., 'trad￾ing' -> 'trading'
    text = replace_numbers(text)
    text = text.translate(str.maketrans("", "", punctuation))  # strip punctuation
    text = re.sub(r"\s+", " ", text)  # standardise spaces
    text = text.lower().strip()
    tokens = []
    for token in text.split():
        if all(
            [
                token.isalnum(),
                1 < len(token) < 35,
                not remove_stopwords or token not in STOP_WORDS,
            ]
        ):
            tokens.append(token)
    text = " ".join(tokens)
    return text


def preprocess_example(example: dict) -> dict:
    example["text"] = replace_numbers(example["text"])
    example["text_clean"] = preprocess_text(example["text"])
    return example
