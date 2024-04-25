"""
Simple CLI for running experiments on SDGi Corpus.
"""

# cli
import click

# data wrangling
from datasets import load_from_disk

# monitoring
import mlflow

# utils
from dotenv import load_dotenv

# local packages
import src


# settings
load_dotenv()
SIZES = ["s", "m", "l", "x"]
LANGUAGES = ["en", "es", "fr", "xx"]
MODELS = {
    "bow_svm": src.run_experiment_bow_svm,
    "ada_svm": src.run_experiment_ada_svm,
    "bow_mlp": src.run_experiment_bow_mlp,
    "ada_mlp": src.run_experiment_ada_mlp,
    "gnn": src.run_experiment_gnn,
    "gpt": src.run_experiment_zeroshot_gpt,
}


@click.command()
@click.option(
    "--size",
    type=click.Choice(SIZES),
    required=True,
    help="Size bucket of the examples.",
)
@click.option(
    "--language",
    type=click.Choice(LANGUAGES),
    required=True,
    help="Language of the examples.",
)
@click.option(
    "--model", type=click.Choice(list(MODELS)), required=True, help="Model type."
)
@click.option(
    "--experiment",
    default="sdgi-corpus",
    help="Experiment name for MLflow run.",
    show_default=True,
)
@click.option(
    "--ood",
    is_flag=True,
    show_default=True,
    default=False,
    help="Use out-of-domain data for avaluation.",
)
def main(size: str, language: str, model: str, experiment: str, ood: bool):
    mlflow.set_experiment(experiment)
    dataset = src.prepare_dataset(size, language)
    experiment_func = MODELS[model]
    run_name = f"sdgi-{size}-{language}-{model}" + ("-ood" if ood else "")
    if ood:
        test_dataset = load_from_disk("data/sdg-meter")
    else:
        test_dataset = None
    if model != "gpt":
        src.log_experiment(
            dataset=dataset,
            experiment_func=experiment_func,
            run_name=run_name,
            test_dataset=test_dataset,
            size=size,
            language=language,
            model=model,
        )
    else:
        MODELS[model](dataset)
    click.echo(dataset.num_rows)
    click.echo("Done!")


if __name__ == "__main__":
    main()
