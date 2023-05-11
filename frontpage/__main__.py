from radicli import Radicli, Arg
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


@cli.command("annotate")
def annotate():
    """Annotate new examples."""
    fp.annotate()


@cli.command("train")
def train():
    """Trains a new model on the data."""
    fp.train()


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
