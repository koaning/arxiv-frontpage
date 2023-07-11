import datetime as dt 
from pathlib import Path 

from jinja2 import Template
from radicli import Radicli

from .download import main as download_data
from .datastream import DataStream
from .modelling import SentenceModel
from .recipe import annotate_prodigy
from .utils import console
from .constants import TEMPLATE_PATH

cli = Radicli()


@cli.command("download")
def download():
    """Download new data."""
    download_data()


@cli.command("index")
def preprocess():
    """Preprocess downloaded data for annotation."""
    DataStream().create_indices(model=SentenceModel())


@cli.command("annotate")
def annotate():
    """Annotate new examples."""
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


@cli.command("build")
def build():
    """Build a new site"""
    # train()
    sections = DataStream().get_site_content()
    console.log("Building site.")
    import srsly
    srsly.write_json("sections.json", sections)
    template = Template(Path(TEMPLATE_PATH).read_text())
    Path("site.html").write_text(template.render(sections=sections, today=dt.date.today()))


@cli.command("benchmark")
def generate():
    """Benchmark the models"""
    pass


if __name__ == "__main__":
    cli.run()
