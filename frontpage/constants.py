import srsly
from pathlib import Path 
from typing import Literal 

from .types import Config

# Paths and folders
INDICES_FOLDER = Path("indices")
TRAINED_FOLDER_FOLDER = Path("training")
TEMPLATE_PATH = Path("templates/home.html")
CONFIG_FILE = "config.yml"

# Possible values
DATA_LEVELS = ["sentence", "abstract"]
DATA_LEVELS_TYPE = Literal["sentence", "abstract"]
CONFIG = Config(**srsly.read_yaml(CONFIG_FILE))
LABELS = [s.label for s in CONFIG.sections]
