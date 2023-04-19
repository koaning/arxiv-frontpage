from radicli import Radicli, Arg
from . import Frontpage

cli = Radicli()

@cli.command("download")
def download():
    """Description of the function for help text."""
    fp = Frontpage.from_config_file("config.yml")
    fp.download()

if __name__ == "__main__":
    cli.run()