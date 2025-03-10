import logging

from lightning.pytorch.utilities.rank_zero import rank_zero_only
from rich.console import Console
from rich.logging import RichHandler
import os

logger = logging.getLogger("yolo")
logger.setLevel(logging.DEBUG)
logger.propagate = False

# allow the user to get a simpler output
# TODO: needs to be better integrated
DISABLE_RICH_HANDLER = bool(os.environ.get('DISABLE_RICH_HANDLER', ''))

if not DISABLE_RICH_HANDLER:
    if rank_zero_only.rank == 0 and not logger.hasHandlers():
        logger.addHandler(RichHandler(console=Console(), show_level=True, show_path=True, show_time=True, markup=True))
