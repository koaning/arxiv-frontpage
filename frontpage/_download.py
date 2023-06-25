import datetime as dt
from pathlib import Path

from wasabi import Printer
import srsly
import arxiv
import itertools as it


def parse_items(items, tag, categories=None, max_age=2):
    for result in items.results():
        if categories:
            print(list(it.product(categories, result.categories)))
            if not any(a in b for a, b in it.product(categories, result.categories)):
                continue

        created = result.published.date()
        limit = dt.date.today() - dt.timedelta(days=max_age)
        if created > limit:
            content = {
                "title": result.title,
                "abstract": str(result.summary).replace("\n", " "),
                "created": str(created)[:10],
                "meta": {
                    "link": result.entry_id,
                    "tag": tag,
                },
            }
            yield content


def main(config):
    """Fetch data from arxiv."""
    msg = Printer()
    downloaded = []
    for section in config["sections"]:
        msg.text(f"Downloading data for {section['name']}", color="cyan")
        for query in section["queries"]:
            items = arxiv.Search(
                query=query["query"],
                max_results=int(10),
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            parsed = list(
                parse_items(
                    items, categories=query.get("section", None), tag=section["tag"]
                )
            )
            msg.text(f"  Found {len(parsed)} items via `{query['query']}`")

            downloaded.extend(list(parsed))

    write_path = Path("downloads") / f"{dt.date.today()}.jsonl"
    srsly.write_jsonl(write_path, downloaded, append=True, append_new_line=False)
    msg.good(f"Written {len(downloaded)} results in `{write_path}`.")


if __name__ == "__main__":
    config = srsly.read_yaml("config.yml")
    main(config)
