import os
import datetime as dt 
from pathlib import Path 

from jinja2 import Template
from radicli import Radicli, Arg

from .utils import console
from .constants import TEMPLATE_PATH, TRAINED_FOLDER, SITE_PATH

cli = Radicli()


@cli.command("download")
def download():
    """Download new data."""
    from .download import main as download_data
    download_data()


@cli.command("index", 
             kind=Arg(help="Can be lunr/simsity"), 
             level=Arg(help="Can be sentence/abstract")
)
def index_cli(kind:str, level:str):
    """Creates index for annotation."""
    from .datastream import DataStream

    DataStream().create_index(level=level, kind=kind)


@cli.command("preprocess")
def preprocess_cli():
    """Dedup and process data for faster processing."""
    from .datastream import DataStream
    DataStream().save_clean_download_stream()


@cli.command("annotate")
def annotate():
    """Annotate new examples."""
    def run_questions():
        import questionary
        from .constants import LABELS, DATA_LEVELS
        results = {}
        results["label"] = questionary.select(
            "Which label do you want to annotate?",
            choices=LABELS,
        ).ask()

        results["level"] = questionary.select(
            "What view of the data do you want to take?",
            choices=DATA_LEVELS,
        ).ask()

        if results["level"] == "abstract":
            choices = ["second-opinion", "search-engine", "simsity", "random"]
        else:
            choices = ["simsity", "search-engine", "active-learning", "random"]

        results["tactic"] = questionary.select(
            "Which tactic do you want to apply?",
            choices=choices,
        ).ask()

        results['setting'] = ''
        if results["tactic"] in ["simsity", "search-engine"]:
            results["setting"] = questionary.text(
                "What query would you like to use?", ""
            ).ask()

        if results["tactic"] == "active-learning":
            results["setting"] = questionary.select(
                "What should the active learning method prefer?",
                choices=["positive class", "uncertainty", "negative class"],
            ).ask()
        return results 
    
    results = run_questions()
    from .recipe import annotate_prodigy
    annotate_prodigy(results)

@cli.command("annotprep")
def annotprep():
    """Prepares data for training."""
    from .datastream import DataStream
    DataStream().save_train_stream()


@cli.command("train")
def train():
    """Trains a new model on the data."""
    from .datastream import DataStream
    from .modelling import SentenceModel
    examples = DataStream().get_train_stream()
    SentenceModel().train(examples=examples).to_disk()


@cli.command("stats")
def stats():
    """Show annotation stats"""
    from .datastream import DataStream
    DataStream().show_annot_stats()


@cli.command(
    "build", 
    retrain=Arg("--retrain", "-rt", help="Retrain model?"),
    prep=Arg("--preprocess", "-pr", help="Preprocess again?")
)
def build(retrain: bool = False, prep:bool = False):
    """Build a new site"""
    from .datastream import DataStream
    if prep:
        preprocess_cli()
    if retrain:
        train()
    console.log("Starting site build process")
    sections = DataStream().get_site_content()
    template = Template(Path(TEMPLATE_PATH).read_text())
    rendered = template.render(sections=sections, today=dt.date.today())
    SITE_PATH.write_text(rendered)
    console.log("Site built.")


@cli.command("artifact",
    action=Arg(help="Can be upload/download"),
)
def artifact(action:str):
    """Upload/download from wandb"""
    import wandb
    from dotenv import load_dotenv
    load_dotenv()
    run = wandb.init(os.getenv("WANDB_API_KEY"))
    if action == "upload":
        artifact = wandb.Artifact('sentence-model', type='model')
        artifact.add_dir(TRAINED_FOLDER)
        run.log_artifact(artifact)
    if action == "download":
        TRAINED_FOLDER.mkdir(exist_ok=True, parents=True)
        artifact = run.use_artifact('sentence-model:latest')
        artifact.download(TRAINED_FOLDER)


@cli.command("benchmark")
def generate():
    """Benchmark the models"""
    pass


if __name__ == "__main__":
    cli.run()
