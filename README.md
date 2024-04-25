# dsc-sdgi-corpus


[![python](https://img.shields.io/badge/Python-3.11-blue.svg?logo=python&logoColor=white)](https://www.python.org)
![Licence](https://img.shields.io/github/license/UNDP-Data/dsc-sdgi-corpus
)
<a href="https://huggingface.co/datasets/UNDP/sdgi-corpus">
    <img alt="Website" src="https://img.shields.io/website?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FUNDP%2Fsdgi-corpus&label=Dataset&up_message=UNDP/sdgi-corpus&up_color=blue">
</a>

## Introduction

Model benchmarks on [SDGi Corpus](https://huggingface.co/datasets/UNDP/sdgi-corpus), a multilingual dataset for text 
classification by Sustainable Development Goals.

## Getting Started

### Python Environment

The codebase has been developed and tested in Python `3.11`. To create a local python environment, clone the repository
and run the following commands in the project directory:

```shell
python -m venv .venv/
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

The following environment variables may need to be set in `.env` file:

```shell
# Location for an MLflow database
MLFLOW_TRACKING_URI="sqlite:///mlruns.db"

# The below is only required for GPT experiments or OOD data
AZURE_OPENAI_API_KEY="<Azure OpenAI API Key>"
AZURE_OPENAI_ENDPOINT="<Azure OpenAI API Endpoint>"
AZURE_OPENAI_EMBEDDING_MODEL="<Azure OpenAI Embedding Model Deployment>"
```

## Running Experiments

For running out-of-domain (OOD) experiments, one needs to first prepare it using a function from `src`. This requires
access Azure OpenAI and setting the env variables mentioned above. To create and save a dataset run:

```python
from src import prepare_ood_dataset

dataset = prepare_ood_dataset()
dataset.save_to_disk("data/sdg-meter")
```

To replicate supervised results from the paper, you can run the Shell script:

```shell
chmod 755 main.sh
./main.sh
```

If you prefer running individual experiments, you can use `main.py`:
```shell
python main.py --size s --language xx
```

Results are saved to a local SQLite database you specify in the enviroment variables. To view the results in MLflow, run:

```shell
mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db
# open http://127.0.0.1:8080
```

## Contribute
If you have any questions or notice any issues, feel free to [open an issue](https://github.com/UNDP-Data/dsc-sdgi-corpus/issues).
