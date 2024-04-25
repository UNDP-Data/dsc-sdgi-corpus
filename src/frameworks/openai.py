# standard library
import os

# wrangling
import numpy as np

# genai
import tiktoken
from openai import AzureOpenAI

__all__ = [
    "get_embeddings",
    "classify_text_zeroshot",
]


SYSTEM_MESSAGE = f"""You are an expert in Sustainable Development Goals (SDGs).
Your task is to classify a text by SDG. You can assign one or more SDGs to a text, but
you should only assign the Goals that are most relevant. Typically, you would assign just one Goal.
You must output only a valid Python list of one-hot encoded labels corresponding to the SDGs.

For example, for a text relevant to Goals 1 and 3, the output would be:

```
[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

for Goals 9, 11 and 12, the output would be:

```
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0]
```

and so on.
"""


def get_client() -> AzureOpenAI:
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-15-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        timeout=10,
    )
    return client


def truncate_text_tokens(text: str, max_tokens: int = 8192) -> list[int]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)[:max_tokens]
    return tokens


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Get OpenAI embeddings for a list of texts.

    Parameters
    ----------
    texts : list[str]
        List of texts to embed.

    Returns
    -------
    embeddings : list[list[float]]
        A list of embeddings for each text.
    """
    client = get_client()
    texts = list(map(truncate_text_tokens, texts))
    response = client.embeddings.create(
        input=texts,
        model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        timeout=10,
    )
    embeddings = [x.embedding for x in response.data]
    return embeddings


def fix_labels(labels) -> list[int]:
    if labels is None or not isinstance(labels, list):
        labels = np.zeros(17)
    elif len(labels) != 17 or any([label not in {0, 1} for label in labels]):
        # e.g., "[3, 10]"
        indices = labels.copy()
        labels = np.zeros(17)
        for index in indices:
            if isinstance(index, int) and 0 < index < 18:
                labels[index - 1] = 1
    try:
        labels = list(map(int, labels))
    except:
        labels = np.zeros(17).tolist()
    return labels


def classify_text_zeroshot(text: str) -> list[int]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    text = encoding.decode(tokens)[: 16384 - 1]
    client = get_client()
    response = client.chat.completions.create(
        model="sdgi-gpt-35-turbo-16k",  # i.e., gpt-35-turbo-16k 0613
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
    )
    content = response.choices[0].message.content
    try:
        labels = eval(content)
    except:
        labels = None
    labels = fix_labels(labels)
    return labels
