import logging
import os
from pathlib import Path

from datetime import datetime

try:
    from from_root import from_root
except ImportError:
    # Fallback to repository root: <root>/us_visa/logger/__init__.py -> parents[2] == <root>
    def from_root() -> str:
        return str(Path(__file__).resolve().parents[2])

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_dir = 'logs'

logs_path = os.path.join(from_root(), log_dir, LOG_FILE)

os.makedirs(os.path.dirname(logs_path), exist_ok=True)


logging.basicConfig(
    filename=logs_path,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)