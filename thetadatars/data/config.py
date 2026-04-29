import os
import json
import platform
from pathlib import Path

ENV_VAR = "THETADATARS_DB"
CONFIG_DIR = "thetadatars"
CONFIG_FILE = "config.json"
DEFAULT_DB = "thetadatars.duckdb"


def _get_config_dir() -> Path:
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")
    elif system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")

    return Path(base) / CONFIG_DIR


def get_db_path() -> Path:
    env_path = os.environ.get(ENV_VAR, "")
    if env_path:
        return Path(env_path)

    config_dir = _get_config_dir()
    config_path = config_dir / CONFIG_FILE

    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            db_path = config.get("database", "")
            if db_path:
                return Path(db_path)
        except (json.JSONDecodeError, OSError):
            pass

    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / DEFAULT_DB
