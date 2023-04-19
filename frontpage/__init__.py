import srsly 
from ._download import main as _download

class Frontpage:
	def __init__(self, config):
		self.config = config

	@classmethod
	def from_config_file(cls, path):
		"""Loads from a config file."""
		return cls(config=srsly.read_yaml(path))

	def download(self):
		"""Download new data for today."""
		_download(self.config)

	def preprocess(self):
		# index all the sentences
		...

	def train(self):
		...

	def annot(self):
		# Start up a query that launched Prodigy with questionary
		...

	def evaluate(self):
		...

	def push_wandb(self):
		...

	def pull_wandb(self):
		...
