import pathlib
from typing import Any

import tomllib

# Caricamento configurazione TOML
config_path: pathlib.Path = pathlib.Path(__file__).parent.parent.parent / "config.toml"
watch_folder: pathlib.Path
output_folder: pathlib.Path
valid_extensions: set[str]
db_path: pathlib.Path

if config_path.exists():
    print(f"Reading config at path: {str(config_path)}")
    with open(config_path, "rb") as f:
        config: dict[str, Any] = tomllib.load(f)

    watch_folder = (
        pathlib.Path(config.get("paths", {}).get("watch_dir", pathlib.Path(__file__)))
        .expanduser()
        .resolve()
    )
    output_folder = (
        pathlib.Path(
            config.get("paths", {}).get("output_dir", pathlib.Path(__file__) / "output")
        )
        .expanduser()
        .resolve()
    )
    db_path = (
        pathlib.Path(
            config.get("db", {}).get(
                "path", pathlib.Path(__file__) / "image_processing.db"
            )
        )
        .expanduser()
        .resolve()
    )
    valid_extensions = set(
        config.get("settings", {}).get("valid_extensions", [".tiff", ".tif"])
    )
    print(f"Config loaded")
else:
    watch_folder = pathlib.Path(__file__)
    output_folder = pathlib.Path(__file__) / "output"
    db_path = pathlib.Path(__file__) / "image_processing.db"
    valid_extensions = {".tiff", ".tif"}

watch_folder.mkdir(exist_ok=True, parents=True)
output_folder.mkdir(exist_ok=True, parents=True)
db_path.parent.mkdir(exist_ok=True, parents=True)
