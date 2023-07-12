import os
import datetime as dt 
from pathlib import Path 

from jinja2 import Template
from radicli import Radicli, Arg

from .download import main as download_data
from .datastream import DataStream
from .modelling import SentenceModel
from .utils import console
from .constants import TEMPLATE_PATH, TRAINED_FOLDER, SITE_PATH

cli = Radicli()


@cli.command("download")
def download():
    """Download new data."""
    download_data()


@cli.command("index")
def index():
    """Preprocess downloaded data for annotation."""
    DataStream().create_indices(model=SentenceModel())


@cli.command("preprocess")
def preprocess():
    """Dedup and process data for faster processing."""
    DataStream().save_clean_download_stream()


@cli.command("annotate")
def annotate():
    """Annotate new examples."""
    from .recipe import annotate_prodigy
    
    annotate_prodigy()


@cli.command("train")
def train():
    """Trains a new model on the data."""
    examples = DataStream().get_train_stream()
    SentenceModel().train(examples=examples).to_disk()


@cli.command("stats")
def stats():
    """Show annotation stats"""
    DataStream().show_annot_stats()


@cli.command(
    "build", 
    retrain=Arg("--retrain", "-rt", help="Retrain model?"),
    prep=Arg("--preprocess", "-pr", help="Preprocess again?")
)
def build(retrain: bool = False, prep:bool = False):
    """Build a new site"""
    if prep:
        preprocess()
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
    run = wandb.init()
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
