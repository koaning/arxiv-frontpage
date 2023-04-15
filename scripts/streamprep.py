from simsity import create_index, load_index
from embetter.text import SentenceEncoder
from simsity.datasets import fetch_recipes

# Load an index from a path
encoder = SentenceEncoder()
reloaded_index = load_index(path="search_index", encoder=encoder)
texts, dists = reloaded_index.query("pork")
