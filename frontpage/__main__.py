from radicli import Radicli
from . import Frontpage

cli = Radicli()
fp = Frontpage.from_config_file("config.yml")


@cli.command("download")
def download():
    """Download new data."""
    fp.download()


@cli.command("preprocess")
def preprocess():
    """Preprocess downloaded data for annotation."""
    fp.preprocess()


@cli.command("index")
def index():
    """Create indices to aid with data annotation."""
    fp.index()


@cli.command("annotate")
def annotate():
    """Annotate new examples."""
    fp.annotate()


@cli.command("train")
def train():
    """Trains a new model on the data."""
    fp.train()


@cli.command("stats")
def stats():
    """Show annotation stats"""
    fp.show_annot_stats()


@cli.command("build")
def build():
    """Build a new site"""
    fp.build()


@cli.command("ptc-gen")
def generate():
    """Creates scripts for Prodigy Teams"""
    fp.teams_create()


@cli.command("explore")
def explore():
    """explores a model."""
    import questionary

    while True:
        mod = fp.model
        sent = questionary.text(
            "Got a sentence?",
        ).ask()
        print(mod(sent))


if __name__ == "__main__":
    cli.run()
