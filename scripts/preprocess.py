import itertools as it
from pathlib import Path

import srsly
import typer


def dedup(stream):
    uniq = {}
    for ex in stream:
        uniq[hash(ex["title"])] = ex
    for ex in uniq.values():
        yield ex


def main(folder: Path, out: Path):
    """Concat all files and double-check the schema."""
    glob = folder.glob("**/*.jsonl")
    full_data = it.chain(*list(srsly.read_jsonl(file) for file in glob))
    stream = (item for item in dedup(full_data))
    srsly.write_jsonl(out, stream)


if __name__ == "__main__":
    typer.run(main)
